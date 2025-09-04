from typing import List
import atexit
import signal
from typing import Optional, Any
from contextlib import suppress

import fire  # type: ignore
from prompt_toolkit import prompt

from codearkt.event_bus import EventType
from codearkt.llm import ChatMessage
from codearkt.util import get_unique_id
from codearkt.client import query_agent, stop_agent
from codearkt.server import DEFAULT_SERVER_PORT, DEFAULT_SERVER_HOST


def main(
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    root_agent_name: str = "manager",
) -> None:
    real_messages: List[ChatMessage] = []
    agent_names: List[str] = []

    session_id = get_unique_id()

    def cleanup_session(signum: Optional[Any] = None, frame: Optional[Any] = None) -> None:
        if session_id:
            with suppress(Exception):
                stop_agent(session_id, host=host, port=port)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()

    atexit.register(cleanup_session)
    signal.signal(signal.SIGINT, cleanup_session)
    signal.signal(signal.SIGTERM, cleanup_session)

    while True:
        message = prompt(
            "Your message (Esc then Enter to accept):\n",
            multiline=True,
        )
        if message == "exit":
            break

        real_messages.append(ChatMessage(role="user", content=message))
        events = query_agent(
            real_messages, session_id=session_id, host=host, port=port, agent_name=root_agent_name
        )
        for event in events:
            is_root_agent = (
                len(agent_names) == 1
                and agent_names[0] == root_agent_name
                and agent_names[0] == event.agent_name
            )
            if event.event_type == EventType.TOOL_RESPONSE:
                if is_root_agent:
                    real_messages.append(
                        ChatMessage(role="user", content="Tool response:\n" + str(event.content))
                    )
                print("Tool Response:\n", event.content)
            elif event.event_type == EventType.AGENT_START:
                print(f"\n**Starting {event.agent_name} agent...**\n\n")
                agent_names.append(event.agent_name)
            elif event.event_type == EventType.AGENT_END:
                print(f"\n**Agent {event.agent_name} completed the task!**\n\n")
                agent_names.pop()
            elif event.event_type == EventType.PLANNING_OUTPUT:
                if not event.content:
                    continue
                print(event.content, end="")
            elif event.event_type == EventType.OUTPUT:
                if not event.content:
                    continue
                print(event.content, end="")
                if is_root_agent:
                    if real_messages[-1].role == "assistant":
                        assert isinstance(real_messages[-1].content, str)
                        real_messages[-1].content += event.content
                    else:
                        real_messages.append(ChatMessage(role="assistant", content=event.content))


if __name__ == "__main__":
    fire.Fire(main)
