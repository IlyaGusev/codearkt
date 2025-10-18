from typing import Any, Dict, List, Union

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.responses import JSONResponse
from starlette.routing import Route

from codearkt.agent_executor import create_multi_agent_a2a_app
from codearkt.codeact import CodeActAgent
from codearkt.event_bus import AgentEventBus
from codearkt.metrics import TokenUsageStore


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
