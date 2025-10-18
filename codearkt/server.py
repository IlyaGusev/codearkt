import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastmcp import settings as fastmcp_settings
from sse_starlette.sse import AppStatus

from codearkt.app_a2a import get_a2a_app
from codearkt.app_mcp import get_mcp_app
from codearkt.codeact import CodeActAgent
from codearkt.event_bus import AgentEventBus
from codearkt.llm import ChatMessage
from codearkt.metrics import TokenUsageStore
from codearkt.settings import settings
from codearkt.util import append_jsonl_atomic, find_free_port, get_unique_id

fastmcp_settings.stateless_http = True


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.05)


def reset_app_status() -> None:
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


def get_main_app(
    agent: CodeActAgent,
    event_bus: AgentEventBus,
    token_usage_store: Optional[TokenUsageStore] = None,
    mcp_config: Optional[Dict[str, Any]] = None,
    server_host: str = settings.DEFAULT_SERVER_HOST,
    server_port: int = settings.DEFAULT_SERVER_PORT,
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_mcp_server_prefixes: bool = True,
) -> FastAPI:
    mcp_app = get_mcp_app(
        mcp_config=mcp_config,
        additional_tools=additional_tools,
        add_prefixes=add_mcp_server_prefixes,
    )

    a2a_app = get_a2a_app(
        agent=agent,
        server_host=server_host,
        server_port=server_port,
        event_bus=event_bus,
        token_usage_store=token_usage_store,
    )

    mcp_app.mount("/a2a", a2a_app)
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
    host: str = settings.DEFAULT_SERVER_HOST,
    port: int = settings.DEFAULT_SERVER_PORT,
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
    host = settings.DEFAULT_SERVER_HOST
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
