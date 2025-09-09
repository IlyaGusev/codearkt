import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator

import uvicorn
from fastmcp import FastMCP, settings as fastmcp_settings
from fastmcp.client.client import Client
from fastmcp.client.transports import (
    SSETransport,
    StreamableHttpTransport,
    ClientTransport,
)
from fastmcp.utilities.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
    infer_transport_type_from_url,
)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import AppStatus

from codearkt.codeact import CodeActAgent
from codearkt.llm import ChatMessage
from codearkt.event_bus import AgentEventBus
from codearkt.util import get_unique_id, find_free_port

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 5055
PROXY_SSE_READ_TIMEOUT = 12 * 60 * 60


fastmcp_settings.stateless_http = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _log(message: str, session_id: str, level: int = logging.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"| {timestamp} | {session_id:<8} | {message}"
    logger.log(level, message)


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.05)


def reset_app_status() -> None:
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


class AgentRequest(BaseModel):  # type: ignore
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    stream: bool = False


class AgentCard(BaseModel):  # type: ignore
    name: str
    description: str


AGENT_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def create_agent_endpoint(
    agent_app: FastAPI,
    agent_instance: CodeActAgent,
    server_host: str,
    server_port: int,
    event_bus: AgentEventBus,
) -> Callable[..., Any]:
    @agent_app.post(f"/{agent_instance.name}")  # type: ignore
    async def agent_tool(request: AgentRequest) -> Any:
        session_id = request.session_id or get_unique_id()

        def _start_agent_task() -> asyncio.Task[Any]:
            task = asyncio.create_task(
                agent_instance.ainvoke(
                    messages=request.messages,
                    session_id=session_id,
                    event_bus=event_bus,
                    server_host=server_host,
                    server_port=server_port,
                )
            )
            event_bus.register_task(
                session_id=session_id,
                agent_name=agent_instance.name,
                task=task,
            )
            return task

        async def stream_response() -> AsyncGenerator[str, None]:
            async for event in event_bus.stream_events(session_id):
                yield event.model_dump_json() + "\n"

        task = _start_agent_task()

        if request.stream:
            return StreamingResponse(
                stream_response(),
                media_type="application/x-ndjson",
                headers=AGENT_RESPONSE_HEADERS,
            )
        else:
            result = await task
            return result

    return agent_tool  # type: ignore


class CancelRequest(BaseModel):  # type: ignore
    session_id: str


def get_agent_app(
    agent: CodeActAgent,
    server_host: str,
    server_port: int,
    event_bus: AgentEventBus,
) -> FastAPI:
    agent_app = FastAPI(
        title="CodeArkt Agent App", description="Agent app for CodeArkt", version="1.0.0"
    )

    agent_cards = []
    for sub_agent in agent.get_all_agents():
        agent_cards.append(AgentCard(name=sub_agent.name, description=sub_agent.description))
        create_agent_endpoint(
            agent_app=agent_app,
            agent_instance=sub_agent,
            server_host=server_host,
            server_port=server_port,
            event_bus=event_bus,
        )

    async def cancel_session(request: CancelRequest) -> Dict[str, str]:
        _log("Finishing session for all agents", request.session_id)
        event_bus.finish_session(request.session_id)
        return {"status": "cancelled", "session_id": request.session_id}

    async def get_agents() -> List[AgentCard]:
        return agent_cards

    agent_app.get("/list")(get_agents)
    agent_app.post("/cancel")(cancel_session)
    return agent_app


def get_mcp_app(
    mcp_config: Optional[Dict[str, Any]],
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_prefixes: bool = True,
) -> FastAPI:
    mcp: FastMCP[Any] = FastMCP(name="Codearkt MCP Proxy")
    if mcp_config:
        cfg = MCPConfig.from_dict(mcp_config)
        server_count = len(cfg.mcpServers)

        for name, server in cfg.mcpServers.items():
            transport: Optional[ClientTransport] = None
            if isinstance(server, RemoteMCPServer):
                transport_type = server.transport or infer_transport_type_from_url(server.url)
                if transport_type == "sse":
                    transport = SSETransport(
                        server.url,
                        headers=server.headers,
                        auth=server.auth,
                        sse_read_timeout=PROXY_SSE_READ_TIMEOUT,
                    )
                else:
                    transport = StreamableHttpTransport(
                        server.url,
                        headers=server.headers,
                        auth=server.auth,
                        sse_read_timeout=PROXY_SSE_READ_TIMEOUT,
                    )
            elif isinstance(server, StdioMCPServer):
                transport = server.to_transport()

            assert transport is not None, "Transport is required for the MCP server in the config"
            client: Client[ClientTransport] = Client(transport=transport)
            sub_proxy = FastMCP.as_proxy(client)
            prefix: Optional[str] = None if server_count == 1 else name
            if not add_prefixes:
                prefix = None
            mcp.mount(prefix=prefix, server=sub_proxy)

    if additional_tools:
        for name, tool in additional_tools.items():
            mcp.tool(tool, name=name)

    return mcp.http_app()


def get_main_app(
    agent: CodeActAgent,
    event_bus: AgentEventBus,
    mcp_config: Optional[Dict[str, Any]] = None,
    server_host: str = DEFAULT_SERVER_HOST,
    server_port: int = DEFAULT_SERVER_PORT,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> FastAPI:
    agent_app = get_agent_app(
        agent=agent,
        server_host=server_host,
        server_port=server_port,
        event_bus=event_bus,
    )
    mcp_app = get_mcp_app(mcp_config, additional_tools, add_prefixes=add_mcp_server_prefixes)
    mcp_app.mount("/agents", agent_app)
    return mcp_app


def run_server(
    agent: CodeActAgent,
    mcp_config: Dict[str, Any],
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> None:
    event_bus = AgentEventBus()
    app = get_main_app(
        agent=agent,
        mcp_config=mcp_config,
        server_host=host,
        server_port=port,
        additional_tools=additional_tools,
        event_bus=event_bus,
        add_mcp_server_prefixes=add_mcp_server_prefixes,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,
        lifespan="on",
        ws="none",
    )


async def _start_temporary_server(
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> tuple[uvicorn.Server, asyncio.Task[None], str, int]:
    event_bus = AgentEventBus()
    host = DEFAULT_SERVER_HOST
    port = find_free_port()
    assert port is not None, "No free port found for temporary server"

    reset_app_status()

    app = get_main_app(
        agent=agent,
        mcp_config=mcp_config,
        server_host=host,
        server_port=port,
        additional_tools=additional_tools,
        event_bus=event_bus,
        add_mcp_server_prefixes=add_mcp_server_prefixes,
    )

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="error",
        access_log=False,
        lifespan="on",
        ws="none",
    )
    server = uvicorn.Server(config)
    server_task: asyncio.Task[None] = asyncio.create_task(server.serve())

    await asyncio.wait_for(_wait_until_started(server), timeout=30)

    return server, server_task, host, port


async def run_query(
    query: str,
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> str:
    server, server_task, host, port = await _start_temporary_server(
        agent,
        mcp_config=mcp_config,
        additional_tools=additional_tools,
        add_mcp_server_prefixes=add_mcp_server_prefixes,
    )

    try:
        result = await agent.ainvoke(
            [ChatMessage(role="user", content=query)],
            session_id=get_unique_id(),
            server_host=host,
            server_port=port,
        )
    finally:
        server.should_exit = True
        await server_task

    return result


async def run_batch(
    queries: List[str],
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 5,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> List[str]:
    if not queries:
        return []

    server, server_task, host, port = await _start_temporary_server(
        agent,
        mcp_config=mcp_config,
        additional_tools=additional_tools,
        add_mcp_server_prefixes=add_mcp_server_prefixes,
    )

    semaphore = asyncio.Semaphore(max_concurrency if max_concurrency > 0 else len(queries))

    async def _run_single(q: str) -> str:
        async with semaphore:
            return await agent.ainvoke(
                [ChatMessage(role="user", content=q)],
                session_id=get_unique_id(),
                server_host=host,
                server_port=port,
            )

    try:
        tasks = [_run_single(q) for q in queries]
        results: List[str] = await asyncio.gather(*tasks)
    finally:
        server.should_exit = True
        await server_task

    return results
