from collections import defaultdict
from typing import Iterator, List, Dict, Any

import gradio as gr
import fire  # type: ignore

from codearkt.event_bus import EventType
from codearkt.llm import ChatMessage
from codearkt.util import get_unique_id
from codearkt.client import query_agent, stop_agent
from codearkt.settings import settings
from codearkt.prompt_storage import (
    DEFAULT_BEGIN_PLAN_SEQUENCE,
    DEFAULT_END_PLAN_SEQUENCE,
    DEFAULT_BEGIN_CODE_SEQUENCE,
    DEFAULT_END_CODE_SEQUENCE,
    DEFAULT_BEGIN_FINAL_ANSWER_SEQUENCE,
    DEFAULT_END_FINAL_ANSWER_SEQUENCE,
)

CODE_TITLE = "Code execution result"


def escape_code_blocks(text: str) -> str:
    text = text.replace(DEFAULT_BEGIN_CODE_SEQUENCE, "```python ")
    text = text.replace(DEFAULT_END_CODE_SEQUENCE, "\n```")
    text = text.replace(DEFAULT_BEGIN_PLAN_SEQUENCE, "")
    text = text.replace(DEFAULT_END_PLAN_SEQUENCE, "")
    text = text.replace(DEFAULT_BEGIN_FINAL_ANSWER_SEQUENCE, "**Final answer**:\n")
    text = text.replace(DEFAULT_END_FINAL_ANSWER_SEQUENCE, "")
    return text


def bot(
    message: str,
    history: List[Dict[str, Any]],
    session_id: str | None,
    real_messages: List[ChatMessage],
    host: str,
    port: int,
) -> Iterator[tuple[List[Dict[str, Any]], str | None, List[ChatMessage]]]:
    history = []
    session_id = session_id or get_unique_id()
    real_messages.append(ChatMessage(role="user", content=message))
    events = query_agent(real_messages, session_id=session_id, host=host, port=port)
    agent_names: List[str] = []
    raw_contents: Dict[int, str] = defaultdict(str)
    history.append({"role": "assistant", "content": ""})
    for event in events:
        session_id = event.session_id or session_id
        prev_message = history[-1]
        prev_message_meta: Dict[str, Any] = prev_message.get("metadata", {})
        prev_message_title = prev_message_meta.get("title")
        is_root_agent = len(agent_names) == 1

        if event.event_type == EventType.TOOL_RESPONSE:
            assert event.content
            if is_root_agent:
                real_messages.append(ChatMessage(role="user", content=event.content))
            history.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "metadata": {"title": CODE_TITLE, "status": "done"},
                }
            )
        elif event.event_type == EventType.AGENT_START:
            agent_names.append(event.agent_name)
            content = f'\n**Starting "{event.agent_name}" agent...**\n\n'
            history.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            raw_contents[len(history) - 1] = content
        elif event.event_type == EventType.AGENT_END:
            last_agent_name = agent_names.pop()
            assert last_agent_name == event.agent_name
            content = f"\n**Agent {event.agent_name} completed the task!**\n\n"
            history.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            raw_contents[len(history) - 1] = content
        elif event.event_type == EventType.OUTPUT or event.event_type == EventType.PLANNING_OUTPUT:
            if prev_message_title == CODE_TITLE:
                history.append({"role": "assistant", "content": ""})

            assert event.content is not None
            current_idx = len(history) - 1
            raw_contents[current_idx] += event.content
            history[-1]["content"] = escape_code_blocks(raw_contents[current_idx])

            if is_root_agent and event.event_type == EventType.OUTPUT:
                if real_messages[-1].role == "assistant" and not real_messages[-1].tool_calls:
                    assert isinstance(real_messages[-1].content, str)
                    real_messages[-1].content += event.content
                else:
                    real_messages.append(ChatMessage(role="assistant", content=event.content))

        yield history, session_id, real_messages


class GradioUI:
    def create_app(self, host: str, port: int) -> Any:
        with gr.Blocks(theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:
            session_id_state = gr.State(None)
            real_messages_state = gr.State([])
            host_state = gr.State(host)
            port_state = gr.State(port)

            chat_iface = gr.ChatInterface(
                bot,
                type="messages",
                additional_inputs=[session_id_state, real_messages_state, host_state, port_state],
                additional_outputs=[session_id_state, real_messages_state],
                save_history=True,
            )

            def _on_stop(session_id: str | None) -> str | None:
                if session_id:
                    stop_agent(session_id, host=host, port=port)
                return session_id

            chat_iface.textbox.stop(_on_stop, inputs=[session_id_state], outputs=[session_id_state])
        return demo

    def run(self, host: str, port: int, share: bool = False) -> None:
        app = self.create_app(host=host, port=port)
        app.queue()
        app.launch(show_error=True, share=share)


def main(
    share: bool = False,
    host: str = settings.DEFAULT_SERVER_HOST,
    port: int = settings.DEFAULT_SERVER_PORT,
) -> None:
    GradioUI().run(share=share, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
