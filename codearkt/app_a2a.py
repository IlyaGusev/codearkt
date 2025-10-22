from typing import Any, Dict, List, Union, Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.responses import JSONResponse
from starlette.routing import Route
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore

from codearkt.codeact import CodeActAgent
from codearkt.event_bus import AgentEventBus
from codearkt.metrics import TokenUsageStore
from codearkt.agent_executor import CodeArktAgentExecutor
from codearkt.agent_executor import create_agent_card


def create_a2a_app_for_agent(
    agent: CodeActAgent,
    event_bus: AgentEventBus,
    token_usage_store: Optional[TokenUsageStore] = None,
    server_host: str = "localhost",
    server_port: int = 8000,
    task_store: Optional[TaskStore] = None,
) -> A2AStarletteApplication:
    server_url = f"http://{server_host}:{server_port}"
    agent_card = create_agent_card(agent, server_url)

    agent_executor = CodeArktAgentExecutor(
        agent=agent,
        event_bus=event_bus,
        token_usage_store=token_usage_store,
        server_host=server_host,
        server_port=server_port,
    )

    if task_store is None:
        task_store = InMemoryTaskStore()

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


def create_multi_agent_a2a_app(
    agents: List[CodeActAgent],
    event_bus: AgentEventBus,
    token_usage_store: Optional[TokenUsageStore] = None,
    server_host: str = "localhost",
    server_port: int = 8000,
) -> Dict[str, A2AStarletteApplication]:
    apps = {}

    for agent in agents:
        app = create_a2a_app_for_agent(
            agent=agent,
            event_bus=event_bus,
            token_usage_store=token_usage_store,
            server_host=server_host,
            server_port=server_port,
        )
        apps[agent.name] = app

    return apps


def get_a2a_app(
    agent: CodeActAgent,
    server_host: str,
    server_port: int,
    event_bus: AgentEventBus,
    token_usage_store: TokenUsageStore | None = None,
    base_path: str = "/a2a",
) -> Starlette:
    all_agents = agent.get_all_agents()

    # For internal agent connections, use localhost instead of 0.0.0.0
    internal_host = "localhost" if server_host == "0.0.0.0" else server_host

    agent_apps = create_multi_agent_a2a_app(
        agents=all_agents,
        event_bus=event_bus,
        token_usage_store=token_usage_store,
        server_host=internal_host,
        server_port=server_port,
    )

    routes: List[Union[Mount, Route]] = []
    for agent_name, a2a_app in agent_apps.items():
        agent_starlette = a2a_app.build()
        mount_path = f"/agents/{agent_name}"
        routes.append(Mount(mount_path, app=agent_starlette))

    # Use the external host for discovery URLs (not localhost)
    external_host = server_host if server_host != "0.0.0.0" else "localhost"

    async def list_agents(request: Any) -> JSONResponse:
        agent_list: List[Dict[str, Any]] = [
            {
                "name": name,
                "url": f"http://{external_host}:{server_port}{base_path}/agents/{name}",
                "agent_card": f"http://{external_host}:{server_port}{base_path}/agents/{name}/.well-known/agent-card.json",
            }
            for name in agent_apps.keys()
        ]
        return JSONResponse({"agents": agent_list})

    routes.insert(0, Route("/agents", list_agents))

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    return Starlette(routes=routes, middleware=middleware)
