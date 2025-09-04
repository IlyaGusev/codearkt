import asyncio
from typing import List

from codearkt.event_bus import AgentEventBus, EventType


async def test_publish_and_stream_events() -> None:
    bus = AgentEventBus()
    session_id = "sess1"
    agent_name = "root_agent"
    bus.register_task(session_id, agent_name, None)

    published_events = [
        (EventType.AGENT_START, None),
        (EventType.OUTPUT, "hello"),
        (EventType.AGENT_END, None),
    ]
    collected: List[EventType] = []

    async def collector() -> None:
        async for event in bus.stream_events(session_id):
            collected.append(event.event_type)
            if event.event_type == EventType.OUTPUT:
                assert event.content == "hello"

    collector_task = asyncio.create_task(collector())
    for event_type, content in published_events:
        await bus.publish_event(session_id, agent_name, event_type, content)
    await collector_task
    assert collected == [evt_type for evt_type, _ in published_events]


async def test_cancel_session_cancels_tasks() -> None:
    bus = AgentEventBus()
    session_id = "sess2"
    agent_name = "agent"

    async def long_running() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(long_running())
    bus.register_task(session_id=session_id, agent_name=agent_name, task=task)
    bus.finish_session(session_id)
    await asyncio.sleep(0)
    assert task.cancelled(), "Task should be cancelled after cancel_session is called."
