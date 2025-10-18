import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    TextPart,
    Part,
    AgentSkill,
)

from codearkt.codeact import CodeActAgent
from codearkt.event_bus import AgentEventBus
from codearkt.llm import ChatMessage
from codearkt.metrics import TokenUsageStore
from codearkt.settings import settings

logger = logging.getLogger(__name__)


class CodeArktAgentExecutor(AgentExecutor):
    def __init__(
        self,
        agent: CodeActAgent,
        event_bus: AgentEventBus,
        token_usage_store: Optional[TokenUsageStore] = None,
        server_host: str = settings.DEFAULT_SERVER_HOST,
        server_port: int = settings.DEFAULT_SERVER_PORT,
    ):

        self.agent = agent
        self.event_bus = event_bus
        self.token_usage_store: Optional[TokenUsageStore] = token_usage_store
        self.server_host = server_host
        self.server_port = server_port
        self.histories: Dict[str, List[ChatMessage]] = defaultdict(list)
        self.running_tasks: Dict[str, asyncio.Task[str]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        assert task_id is not None
        context_id = context.context_id
        assert context_id is not None
        updater = TaskUpdater(event_queue, task_id, context_id)
        message_content = self._extract_message_content(context)
        self.histories[context_id].append(ChatMessage(role="user", content=message_content))

        def _start_agent_task() -> asyncio.Task[Any]:
            task = asyncio.create_task(
                self.agent.ainvoke(
                    messages=self.histories[context_id],
                    session_id=context_id,
                    event_bus=self.event_bus,
                    token_usage_store=self.token_usage_store,
                    server_host=self.server_host,
                    server_port=self.server_port,
                )
            )
            self.event_bus.register_task(
                session_id=context_id,
                agent_name=self.agent.name,
                task=task,
            )
            return task

        task = _start_agent_task()
        self.running_tasks[task_id] = task
        result = await task
        self.histories[context_id].append(ChatMessage(role="assistant", content=result))
        await updater.add_artifact(
            [Part(root=TextPart(text=result))],
            name="final_result",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        assert task_id is not None
        context_id = context.context_id
        assert context_id is not None
        updater = TaskUpdater(event_queue, task_id, context_id)
        agent_task = self.running_tasks.get(task_id)
        if agent_task and not agent_task.done():
            agent_task.cancel()
        await updater.cancel()

    def _extract_message_content(self, context: RequestContext) -> str:
        if not context.message:
            return ""

        message_parts = context.message.parts
        if not message_parts:
            return ""

        text_parts = []
        for part in message_parts:
            if isinstance(part, TextPart):
                text_parts.append(part.text)
            elif hasattr(part, "text"):
                text_parts.append(part.text)
            elif isinstance(part, dict) and part.get("kind") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, Part) and isinstance(part.root, TextPart):
                text_parts.append(part.root.text)

        return "\n".join(text_parts)


def create_agent_card(
    agent: CodeActAgent,
    server_url: str,
) -> AgentCard:
    skills: List[AgentSkill] = []
    capabilities = AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateTransitionHistory=True,
    )

    return AgentCard(
        name=agent.name,
        description=agent.description,
        url=server_url,
        version="1.0.0",
        capabilities=capabilities,
        skills=skills,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )


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
