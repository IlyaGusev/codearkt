import asyncio
import functools
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import anyio
import httpx
from mcp import ClientSession, Tool
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ContentBlock
from pydantic import ValidationError

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 24 * 60 * 60))
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", 12 * 60 * 60))
SERVER_URL_TEMPLATE = os.getenv("SERVER_URL_TEMPLATE", "http://host.docker.internal:{port}")


ToolReturnType = Any

_tool_schemas: Dict[str, Tool] = {}


async def _message_handler(session: ClientSession, exc: Exception) -> None:
    """
    Custom message handler for ValidationError.
    See https://github.com/modelcontextprotocol/python-sdk/issues/1144.
    """
    await anyio.lowlevel.checkpoint()
    if isinstance(exc, ValidationError):
        session._task_group.cancel_scope.cancel()


def _compose_arguments(tool_schema: Optional[Tool], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    arguments = dict(kwargs)
    if not args:
        return arguments
    has_input_schema = (
        tool_schema is not None
        and hasattr(tool_schema, "inputSchema")
        and tool_schema.inputSchema
        and isinstance(tool_schema.inputSchema, dict)
        and "properties" in tool_schema.inputSchema
    )
    if not has_input_schema:
        return arguments

    assert tool_schema
    param_names = list(tool_schema.inputSchema["properties"].keys())
    for i, arg in enumerate(args):
        if i >= len(param_names):
            break
        param_name = param_names[i]
        if param_name not in arguments:
            arguments[param_name] = arg
    return arguments


async def _acall(tool: str, tool_server_port: int, *args: Any, **kwargs: Any) -> ToolReturnType:
    base_url = SERVER_URL_TEMPLATE.format(port=tool_server_port)
    async with streamablehttp_client(
        base_url + "/mcp",
        timeout=TOOL_TIMEOUT,
        sse_read_timeout=TOOL_TIMEOUT,
    ) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            session._message_handler = partial(_message_handler, session)
            tool_schema = _tool_schemas.get(tool)
            arguments = _compose_arguments(tool_schema, *args, **kwargs)

            result: CallToolResult = await session.call_tool(tool, arguments)

            structured_content = result.structuredContent
            if structured_content:
                if "result" in structured_content and len(structured_content) == 1:
                    return structured_content["result"]
                return structured_content

            content_blocks: List[ContentBlock] = result.content
            if len(content_blocks) == 0:
                return None
            if len(content_blocks) == 1 and content_blocks[0].type == "text":
                return content_blocks[0].text
            return content_blocks


def _call(tool: str, tool_server_port: int, *args: Any, **kwargs: Any) -> ToolReturnType:
    def runner() -> ToolReturnType:
        return asyncio.run(_acall(tool, tool_server_port, *args, **kwargs))

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(runner)
        return future.result(timeout=TOOL_TIMEOUT + 5)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def fetch_tools(
    tool_server_port: Optional[int] = None,
) -> Dict[str, Callable[..., ToolReturnType]]:
    if not tool_server_port:
        return {}
    global _tool_schemas
    final_tools = {}
    base_url = SERVER_URL_TEMPLATE.format(port=tool_server_port)
    print("Tools server URL", base_url)
    try:
        async with streamablehttp_client(base_url + "/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                tools: List[Tool] = tools_response.tools

                _tool_schemas = {tool.name: tool for tool in tools}

                for tool in tools:
                    tool_fn: Callable[..., ToolReturnType] = functools.partial(
                        _call, tool.name, tool_server_port
                    )
                    final_tools[tool.name] = tool_fn
    except Exception:
        print("Failed to fetch MCP tools")
        print(traceback.format_exc())
        pass

    agent_cards = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(base_url + "/agents/list")
            response.raise_for_status()
            agent_cards = response.json()
    except Exception:
        print("Failed to fetch agents")
        print(traceback.format_exc())
        pass

    for card in agent_cards:
        agent_name = card["name"]
        url = base_url + f"/agents/{agent_name}"

        def create_call_agent(url: str) -> Callable[..., Any]:
            def _call_agent(query: str, session_id: str) -> Any:
                payload = {
                    "messages": [{"role": "user", "content": query}],
                    "session_id": session_id,
                    "stream": False,
                }
                resp = httpx.post(url, json=payload, timeout=AGENT_TIMEOUT)
                resp.raise_for_status()
                return resp.json()

            return _call_agent

        final_tools["agent__" + agent_name] = create_call_agent(url)

    return final_tools
