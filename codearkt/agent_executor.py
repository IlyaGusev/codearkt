import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
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
    ) -> None:

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
        user_message = ChatMessage(role="user", content=message_content)
        self.histories[context_id].append(user_message)

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

        async def stream_handler() -> None:
            async for event in self.event_bus.stream_events(context_id):
                message = updater.new_agent_message(
                    [Part(root=TextPart(text=event.model_dump_json()))],
                    metadata={"is_event_bus_event": True},
                )
                if not event_queue.is_closed():
                    await event_queue.enqueue_event(message)

        stream_task = asyncio.create_task(stream_handler())
        final_result, _ = await asyncio.gather(task, stream_task)

        result_message = ChatMessage(role="assistant", content=final_result)
        self.histories[context_id].append(result_message)
        await updater.add_artifact(
            [Part(root=TextPart(text=final_result))],
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
