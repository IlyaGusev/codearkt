import asyncio
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator

import uvicorn
from fastmcp import FastMCP, settings as fastmcp_settings
from fastmcp.client.transports import (
    SSETransport,
    StreamableHttpTransport,
    ClientTransport,
)
from fastmcp.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
    infer_transport_type_from_url,
)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import AppStatus
import contextlib

from codearkt.codeact import CodeActAgent
from codearkt.llm import ChatMessage
from codearkt.event_bus import AgentEventBus
from codearkt.metrics import TokenUsageStore
from codearkt.util import get_unique_id, find_free_port, append_jsonl_atomic

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
    token_usage_store: TokenUsageStore | None = None,
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
                    token_usage_store=token_usage_store,
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
    token_usage_store: TokenUsageStore | None = None,
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
            token_usage_store=token_usage_store,
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
            sub_proxy = FastMCP.as_proxy(backend=transport)
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
    token_usage_store: Optional[TokenUsageStore] = None,
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
        token_usage_store=token_usage_store,
    )
    mcp_app = get_mcp_app(mcp_config, additional_tools, add_prefixes=add_mcp_server_prefixes)
    mcp_app.mount("/agents", agent_app)
    return mcp_app


async def _shutdown_server(
    server: uvicorn.Server,
    server_task: asyncio.Task[None],
    timeout: float = 10.0,
) -> None:
    AppStatus.should_exit = True
    server.should_exit = True

    try:
        await asyncio.wait_for(server_task, timeout=timeout / 2.0)
        return
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    server.force_exit = True
    try:
        await asyncio.wait_for(server_task, timeout=timeout / 2.0)
        return
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    with contextlib.suppress(asyncio.CancelledError):
        server_task.cancel()
        await server_task


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
) -> tuple[uvicorn.Server, asyncio.Task[None], str, int, TokenUsageStore]:
    event_bus = AgentEventBus()
    token_usage_store = TokenUsageStore()
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
        token_usage_store=token_usage_store,
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
    server_task: asyncio.Task[None] = asyncio.create_task(server._serve(sockets=None))

    await asyncio.wait_for(_wait_until_started(server), timeout=30)

    return server, server_task, host, port, token_usage_store


async def run_query(
    query: str,
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> str:
    server, server_task, host, port, _ = await _start_temporary_server(
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
        await _shutdown_server(server, server_task)

    return result


async def run_batch(
    queries: List[str],
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 5,
    task_timeout: Optional[int] = None,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
    output_path: Optional[Path] = None,
) -> List[str]:
    if not queries:
        return []

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    server, server_task, host, port, token_usage_store = await _start_temporary_server(
        agent,
        mcp_config=mcp_config,
        additional_tools=additional_tools,
        add_mcp_server_prefixes=add_mcp_server_prefixes,
    )

    semaphore = asyncio.Semaphore(max_concurrency if max_concurrency > 0 else len(queries))

    async def _run_single(q: str) -> str:
        async with semaphore:
            start_time = time.time()
            session_id = get_unique_id()
            task = asyncio.create_task(
                agent.ainvoke(
                    [ChatMessage(role="user", content=q)],
                    session_id=session_id,
                    server_host=host,
                    server_port=port,
                    token_usage_store=token_usage_store,
                )
            )
            result: str
            try:
                if task_timeout and task_timeout > 0:
                    result = await asyncio.wait_for(task, timeout=task_timeout)
                else:
                    result = await task
            except asyncio.CancelledError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(task, timeout=2)
                result = f"Timeout after {task_timeout}"
            except Exception as e:
                result = f"Error: {e}"
            finally:
                end_time = time.time()
                duration = int(end_time - start_time)
                if output_path:
                    token_usage = token_usage_store.get(session_id).model_dump()
                    append_jsonl_atomic(
                        output_path,
                        {
                            "query": q,
                            "result": result,
                            "session_id": session_id,
                            "token_usage": token_usage,
                            "duration": duration,
                        },
                    )
                return result

    results: List[str] = []
    try:
        tasks: List[asyncio.Task[str]] = [asyncio.create_task(_run_single(q)) for q in queries]
        try:
            results = await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for t in tasks:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(_shutdown_server(server, server_task))

    return results
