import asyncio
from typing import Any, Dict, Optional, AsyncGenerator, List
from enum import StrEnum
from collections import defaultdict

from pydantic import BaseModel

STREAM_TIMEOUT = 24 * 60 * 60  # 24 hours


class EventType(StrEnum):
    AGENT_START = "agent_start"
    OUTPUT = "output"
    PLANNING_OUTPUT = "planning_output"
    TOOL_RESPONSE = "observation"
    AGENT_END = "agent_end"


class AgentEvent(BaseModel):  # type: ignore
    session_id: str
    agent_name: str
    event_type: EventType
    content: Optional[str] = None


class AgentEventBus:
    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue[AgentEvent]] = {}
        self.running_tasks: Dict[str, List[asyncio.Task[Any]]] = defaultdict(list)
        self.root_agent_name: Dict[str, str] = {}

    def register_task(
        self, session_id: str, agent_name: str, task: asyncio.Task[Any] | None
    ) -> None:
        if session_id not in self.running_tasks:
            self.root_agent_name[session_id] = agent_name
        if task is not None:
            self.running_tasks[session_id].append(task)

    def finish_session(self, session_id: str) -> None:
        tasks = self.running_tasks.pop(session_id, [])
        while tasks:
            task = tasks.pop()
            task.cancel()
        self.root_agent_name.pop(session_id, None)
        self.queues.pop(session_id, None)

    async def publish_event(
        self,
        session_id: str,
        agent_name: str,
        event_type: EventType = EventType.OUTPUT,
        content: Optional[str] = None,
    ) -> None:
        event = AgentEvent(
            session_id=session_id,
            agent_name=agent_name,
            event_type=event_type,
            content=content,
        )
        if event.session_id not in self.queues:
            self.queues[event.session_id] = asyncio.Queue()
        await self.queues[event.session_id].put(event)

    async def stream_events(self, session_id: str) -> AsyncGenerator[AgentEvent, None]:
        if session_id not in self.queues:
            self.queues[session_id] = asyncio.Queue()
        queue = self.queues[session_id]
        assert session_id in self.root_agent_name, f"Session {session_id} is not tied to an agent"
        root_agent_name = self.root_agent_name[session_id]
        is_agent_end = False
        is_root_agent = True
        while not is_agent_end or not is_root_agent:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=STREAM_TIMEOUT)
                if not event:
                    continue
                yield event
                is_agent_end = event.event_type == EventType.AGENT_END
                is_root_agent = event.agent_name == root_agent_name
            except asyncio.TimeoutError:
                break
