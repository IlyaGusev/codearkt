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


_session_id: str | None = None


def cleanup_session(signum: Optional[Any] = None, frame: Optional[Any] = None) -> None:
    if _session_id:
        with suppress(Exception):
            stop_agent(_session_id)
    if signum == signal.SIGINT:
        raise KeyboardInterrupt()


atexit.register(cleanup_session)
signal.signal(signal.SIGINT, cleanup_session)
signal.signal(signal.SIGTERM, cleanup_session)


def main(host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT) -> None:
    real_messages: List[ChatMessage] = []
    agent_names: List[str] = []
    global _session_id
    while True:
        message = prompt(
            "Your message (Esc then Enter to accept):\n",
            multiline=True,
        )
        if message == "exit":
            break
        _session_id = _session_id or get_unique_id()
        real_messages.append(ChatMessage(role="user", content=message))
        events = query_agent(real_messages, session_id=_session_id, host=host, port=port)
        for event in events:
            is_root_agent = len(agent_names) == 1
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
