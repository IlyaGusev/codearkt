import os
import anyio
import asyncio
import functools
import httpx
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any, Optional
from functools import partial

from pydantic import ValidationError

from mcp import ClientSession, Tool
from mcp.types import ContentBlock
from mcp.client.streamable_http import streamablehttp_client

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 24 * 60 * 60))
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", 12 * 60 * 60))
SERVER_URL_TEMPLATE = "http://host.docker.internal:{port}"


ToolReturnType = List[ContentBlock] | str | None

_tool_schemas: Dict[str, Tool] = {}


async def _message_handler(session: ClientSession, exc: Exception) -> None:
    """
    Custom message handler for ValidationError.
    See https://github.com/modelcontextprotocol/python-sdk/issues/1144.
    """
    await anyio.lowlevel.checkpoint()
    if isinstance(exc, ValidationError):
        session._task_group.cancel_scope.cancel()


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
            arguments = dict(kwargs)
            if args and tool in _tool_schemas:
                tool_schema = _tool_schemas[tool]
                if hasattr(tool_schema, "inputSchema") and tool_schema.inputSchema:
                    if (
                        isinstance(tool_schema.inputSchema, dict)
                        and "properties" in tool_schema.inputSchema
                    ):
                        param_names = list(tool_schema.inputSchema["properties"].keys())
                    else:
                        param_names = []
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            param_name = param_names[i]
                            if param_name not in arguments:
                                arguments[param_name] = arg

            result = await session.call_tool(tool, arguments)
            content_blocks: List[ContentBlock] = result.content
            if len(content_blocks) == 0:
                return None
            if len(content_blocks) == 1 and content_blocks[0].type == "text":
                return content_blocks[0].text
            return content_blocks


def _call(
    tool: str, tool_server_port: int, *args: Any, **kwargs: Any
) -> List[ContentBlock] | str | None:
    def runner() -> List[ContentBlock] | str | None:
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
