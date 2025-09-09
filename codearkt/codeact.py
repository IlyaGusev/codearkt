import re
import copy
import logging
import traceback
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Self, Dict, Any, Optional, Sequence, get_args

import yaml

from mcp import Tool
from jinja2 import Template

from codearkt.python_executor import PythonExecutor
from codearkt.tools import fetch_tools
from codearkt.event_bus import AgentEventBus, EventType
from codearkt.llm import LLM, ChatMessages, ChatMessage
from codearkt.util import get_unique_id, truncate_content

DEFAULT_END_CODE_SEQUENCE = "<end_code>"
DEFAULT_END_PLAN_SEQUENCE = "<end_plan>"
DEFAULT_STOP_SEQUENCES = [DEFAULT_END_CODE_SEQUENCE, "Observation:", "Calling tools:"]
AGENT_TOOL_PREFIX = "agent__"
DEFAULT_MAX_ITERATIONS = 20
PLANNING_LAST_N = 50
PLANNING_CONTENT_MAX_LENGTH = 4000


def extract_code_from_text(text: str) -> str | None:
    pattern = r"[Cc]ode[\*]*\:[\*]*\s*\n*```(?:py|python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


@dataclass
class Prompts:
    system: Template
    final: Template
    no_code_action: Template
    plan: Optional[Template] = None
    plan_prefix: Optional[Template] = None
    plan_suffix: Optional[Template] = None
    end_code_sequence: str = DEFAULT_END_CODE_SEQUENCE
    end_plan_sequence: str = DEFAULT_END_PLAN_SEQUENCE
    stop_sequences: List[str] = field(default_factory=lambda: DEFAULT_STOP_SEQUENCES)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            template = f.read()
        templates: Dict[str, Any] = yaml.safe_load(template)
        wrapped_templates: Dict[str, Any] = {}
        for key, value in templates.items():
            if value is None:
                continue
            field_type = cls.__annotations__.get(key)
            if field_type is Template or Template in get_args(field_type):
                wrapped_templates[key] = Template(value)
            elif field_type is str:
                wrapped_templates[key] = value.strip()
            else:
                wrapped_templates[key] = value
        obj = cls(**wrapped_templates)
        obj.stop_sequences = [s.strip() for s in obj.stop_sequences]
        return obj

    @classmethod
    def default(cls) -> Self:
        current_dir = Path(__file__).parent
        return cls.load(current_dir / "prompts" / "default.yaml")


class CodeActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        tool_names: Sequence[str] = tuple(),
        prompts: Optional[Prompts] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbosity_level: int = logging.ERROR,
        planning_interval: Optional[int] = None,
        managed_agents: Optional[List[Self]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.llm: LLM = llm
        self.prompts: Prompts = prompts or Prompts.default()
        self.tool_names = list(tool_names)
        self.max_iterations = max_iterations
        self.verbosity_level = verbosity_level
        self.planning_interval = planning_interval
        self.managed_agents: Optional[List[Self]] = managed_agents

        if self.managed_agents:
            for agent in self.managed_agents:
                agent_tool_name = AGENT_TOOL_PREFIX + agent.name
                if agent_tool_name not in self.tool_names:
                    self.tool_names.append(agent_tool_name)

        self.logger = logging.getLogger(self.__class__.__name__ + ":" + self.name)
        self.logger.setLevel(self.verbosity_level)

    def get_all_agents(self) -> List[Self]:
        agents = [self]
        if self.managed_agents:
            agents.extend(self.managed_agents)
            for agent in self.managed_agents:
                agents.extend(agent.get_all_agents())
        named_agents = {agent.name: agent for agent in agents}
        return list(named_agents.values())

    async def ainvoke(
        self,
        messages: ChatMessages,
        session_id: str,
        event_bus: AgentEventBus | None = None,
        server_host: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> str:
        messages = copy.deepcopy(messages)

        run_id = get_unique_id()
        await self._publish_event(event_bus, session_id, EventType.AGENT_START)
        self._log(f"Starting agent {self.name}", run_id=run_id, session_id=session_id)

        python_executor = None
        try:
            python_executor = PythonExecutor(
                session_id=session_id,
                tool_names=self.tool_names,
                interpreter_id=run_id,
                tools_server_port=server_port,
                tools_server_host=server_host,
            )
            self._log("Python interpreter started", run_id=run_id, session_id=session_id)
            self._log(
                f"Host: {server_host}, port: {server_port}",
                run_id=run_id,
                session_id=session_id,
            )

            tools = await self._get_tools(server_host=server_host, server_port=server_port)
            self._log(
                f"Fetched tools: {[tool.name for tool in tools]}",
                run_id=run_id,
                session_id=session_id,
            )
            current_date = datetime.now().strftime("%Y-%m-%d")
            system_prompt = self.prompts.system.render(tools=tools, current_date=current_date)

            if messages and messages[0].role not in ("system", "developer"):
                messages = [ChatMessage(role="system", content=system_prompt)] + messages

            for step_number in range(1, self.max_iterations + 1):
                # Optional planning step
                if self.planning_interval is not None and (
                    step_number == 1 or (step_number - 1) % self.planning_interval == 0
                ):
                    self._log(
                        f"Planning step {step_number} started",
                        run_id=run_id,
                        session_id=session_id,
                    )
                    new_messages = await self._run_planning_step(
                        messages=messages,
                        tools=tools,
                        run_id=run_id,
                        session_id=session_id,
                        event_bus=event_bus,
                    )
                    messages.extend(new_messages)
                    self._log(
                        f"Planning step {step_number} completed",
                        run_id=run_id,
                        session_id=session_id,
                    )

                # Main step
                self._log(
                    f"Step {step_number} started",
                    run_id=run_id,
                    session_id=session_id,
                )
                new_messages = await self._step(
                    messages,
                    python_executor=python_executor,
                    session_id=session_id,
                    run_id=run_id,
                    event_bus=event_bus,
                    step_number=step_number,
                )
                messages.extend(new_messages)
                self._log(
                    f"Step {step_number} completed",
                    run_id=run_id,
                    session_id=session_id,
                )
                if messages[-1].role == "assistant":
                    break
            else:
                # Final step
                new_messages = await self._handle_final_message(
                    messages, session_id=session_id, run_id=run_id, event_bus=event_bus
                )
                messages.extend(new_messages)
                self._log(
                    "Final step completed",
                    run_id=run_id,
                    session_id=session_id,
                )

        except Exception as e:
            self._log(
                f"Agent {self.name} failed with error: {e}",
                run_id=run_id,
                session_id=session_id,
                level=logging.ERROR,
            )
            raise e
        finally:
            # Cleanup
            if python_executor:
                await python_executor.cleanup()
            await self._publish_event(event_bus, session_id, EventType.AGENT_END)

        self._log(f"Agent {self.name} completed successfully", run_id=run_id, session_id=session_id)
        return str(messages[-1].content)

    async def _get_tools(
        self,
        server_host: Optional[str],
        server_port: Optional[int],
    ) -> List[Tool]:
        tools = []
        fetched_tool_names = []
        if server_host and server_port:
            server_url = f"{server_host}:{server_port}"
            tools = await fetch_tools(server_url)
            for tool in tools:
                if tool.description:
                    tool.description = dedent(tool.description).strip()
            tools = [tool for tool in tools if tool.name in self.tool_names]
            fetched_tool_names = [tool.name for tool in tools]

        for tool_name in self.tool_names:
            assert (
                tool_name in fetched_tool_names
            ), f"Tool {tool_name} not found in {fetched_tool_names}"
        return tools

    async def _step(
        self,
        messages: ChatMessages,
        python_executor: PythonExecutor,
        session_id: str,
        run_id: str,
        event_bus: AgentEventBus | None = None,
        step_number: int | None = None,
    ) -> ChatMessages:
        _ = step_number  # for logs only
        self._log(
            f"Step inputs: {messages}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        self._log("LLM generates outputs...", run_id=run_id, session_id=session_id)
        output_text = ""
        try:
            output_stream = self.llm.astream(messages, stop=self.prompts.stop_sequences)
            async for event in output_stream:
                delta = event.choices[0].delta
                if isinstance(delta.content, str):
                    chunk = delta.content
                elif isinstance(delta.content, list):
                    chunk = "\n".join([str(item) for item in delta.content])
                output_text += chunk
                await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)
                if any(
                    stop_sequence in output_text for stop_sequence in self.prompts.stop_sequences
                ):
                    break
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, "\n")

        except Exception:
            exception = traceback.format_exc()
            error_text = f"LLM failed with error: {exception}. Please try again."
            self._log(error_text, run_id=run_id, session_id=session_id, level=logging.ERROR)
            return []

        if (
            output_text
            and output_text.strip().endswith("```")
            and not output_text.strip().endswith(self.prompts.end_code_sequence)
        ):
            chunk = self.prompts.end_code_sequence + "\n"
            output_text += chunk

        for stop_sequence in self.prompts.stop_sequences:
            if stop_sequence in output_text:
                output_text = output_text.split(stop_sequence)[0].strip()
                if stop_sequence == self.prompts.end_code_sequence:
                    output_text += stop_sequence
                break

        self._log(
            f"Step output: {output_text}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        self._log("LLM generated outputs!", run_id=run_id, session_id=session_id)

        # Code detection
        code_action = extract_code_from_text(output_text)

        # No code action
        new_messages = []
        if code_action is None:
            self._log("No tool calls detected", run_id=run_id, session_id=session_id)
            new_messages.append(ChatMessage(role="assistant", content=output_text))
            if self._is_final_answer(output_text):
                self._log("Final answer found!", run_id=run_id, session_id=session_id)
                return new_messages
            assert self.prompts.no_code_action is not None
            no_code_action_prompt = self.prompts.no_code_action.render()
            new_messages.append(ChatMessage(role="user", content=no_code_action_prompt))
            self._log(
                no_code_action_prompt,
                run_id=run_id,
                session_id=session_id,
                level=logging.DEBUG,
            )
            return new_messages

        self._log(
            f"Code action: {code_action}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        tool_call_message = ChatMessage(
            role="assistant",
            content=output_text,
        )
        new_messages.append(tool_call_message)

        # Execute code
        try:
            self._log("Executing code...", run_id=run_id, session_id=session_id)
            code_result = await python_executor.ainvoke(code_action)
            self._log(
                f"Code result: {code_result}",
                run_id=run_id,
                session_id=session_id,
                level=logging.DEBUG,
            )
            code_result_message: ChatMessage = code_result.to_message()
            assert isinstance(code_result_message.content, list)
            new_messages.append(code_result_message)
            tool_output: str = str(code_result_message.content[0]["text"]) + "\n"
            await self._publish_event(event_bus, session_id, EventType.TOOL_RESPONSE, tool_output)
            self._log("Code was executed!", run_id=run_id, session_id=session_id)
        except Exception as e:
            new_messages.append(ChatMessage(role="user", content=f"Error: {e}"))
            self._log(f"Code error: {e}", run_id=run_id, session_id=session_id, level=logging.DEBUG)
            await self._publish_event(
                event_bus, session_id, EventType.TOOL_RESPONSE, f"Error: {e}\n"
            )
        return new_messages

    def _is_final_answer(self, content: str) -> bool:
        code_action = extract_code_from_text(content)
        assert code_action is None
        content = content.lower()
        return "final answer:" in content

    async def _handle_final_message(
        self,
        messages: ChatMessages,
        session_id: str,
        run_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        prompt: str = self.prompts.final.render()
        final_message = ChatMessage(role="user", content=prompt)
        input_messages = messages + [final_message]
        self._log(
            f"Final input messages: {input_messages}",
            run_id=run_id,
            session_id=session_id,
            level=logging.DEBUG,
        )

        output_stream = self.llm.astream(input_messages, stop=self.prompts.stop_sequences)
        output_text = ""
        async for event in output_stream:
            delta = event.choices[0].delta
            if isinstance(delta.content, str):
                chunk = delta.content
            elif isinstance(delta.content, list):
                chunk = "\n".join([str(item) for item in delta.content])
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)

        self._log(
            f"Final message: {final_message}",
            run_id=run_id,
            session_id=session_id,
            level=logging.DEBUG,
        )

        return [ChatMessage(role="assistant", content=output_text)]

    def _process_messages_for_planning(
        self,
        messages: ChatMessages,
        last_n: int = PLANNING_LAST_N,
        content_max_length: int = PLANNING_CONTENT_MAX_LENGTH,
    ) -> str:
        messages = copy.deepcopy(messages)

        def messages_to_string(messages_internal: ChatMessages) -> str:
            str_messages = []
            for m in messages_internal:
                content = truncate_content(str(m.content), max_length=content_max_length)
                str_messages.append(f"{m.role}: {content}")
            return "\n\n".join(str_messages)

        assert self.prompts.plan_prefix is not None
        assert self.prompts.plan_suffix is not None
        plan_prefix = self.prompts.plan_prefix.render().strip()
        plan_suffix = self.prompts.plan_suffix.render().strip()
        used_messages = []
        for message in messages:
            if message.role == "system":
                continue
            content = str(message.content)
            if plan_prefix in content or plan_suffix in content:
                continue
            used_messages.append(message)
        if not used_messages:
            return ""
        if len(used_messages) <= last_n:
            return messages_to_string(used_messages)
        conversation = messages_to_string(used_messages[-last_n:])
        first_message = messages_to_string(used_messages[:1])
        return f"First message:\n\n{first_message}\n\nLast {last_n} messages:\n\n{conversation}"

    async def _run_planning_step(
        self,
        messages: ChatMessages,
        tools: List[Tool],
        run_id: str,
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        messages = copy.deepcopy(messages)
        assert self.prompts.plan is not None, "Planning prompt is not set, but planning is enabled"
        assert (
            self.prompts.plan_prefix is not None
        ), "Plan prefix is not set, but planning is enabled"
        assert (
            self.prompts.plan_suffix is not None
        ), "Plan suffix is not set, but planning is enabled"

        conversation = self._process_messages_for_planning(messages)
        current_date = datetime.now().strftime("%Y-%m-%d")
        planning_prompt = self.prompts.plan.render(
            conversation=conversation, tools=tools, current_date=current_date
        )
        input_messages = [ChatMessage(role="user", content=planning_prompt)]

        try:
            output_stream = self.llm.astream(input_messages, stop=[self.prompts.end_plan_sequence])

            plan_prefix = self.prompts.plan_prefix.render().strip() + "\n\n"
            await self._publish_event(event_bus, session_id, EventType.PLANNING_OUTPUT, plan_prefix)
            output_text = plan_prefix

            async for event in output_stream:
                delta = event.choices[0].delta
                chunk = delta.content
                if chunk is None:
                    continue
                assert isinstance(chunk, str), event
                output_text += chunk
                await self._publish_event(event_bus, session_id, EventType.PLANNING_OUTPUT, chunk)
                if self.prompts.end_plan_sequence in output_text:
                    break

            if self.prompts.end_plan_sequence in output_text:
                output_text = output_text.split(self.prompts.end_plan_sequence)[0].strip()
                output_text += self.prompts.end_plan_sequence

            plan_suffix = self.prompts.plan_suffix.render().strip()
            return [
                ChatMessage(role="assistant", content=output_text),
                ChatMessage(role="user", content=plan_suffix),
            ]

        except Exception:
            exception = traceback.format_exc()
            error_text = f"LLM failed with error: {exception}. Please try again."
            self._log(error_text, run_id=run_id, session_id=session_id, level=logging.ERROR)
            return []

    async def _publish_event(
        self,
        event_bus: AgentEventBus | None,
        session_id: str,
        event_type: EventType = EventType.OUTPUT,
        content: Optional[str] = None,
    ) -> None:
        if not event_bus:
            return
        await event_bus.publish_event(
            session_id=session_id,
            agent_name=self.name,
            event_type=event_type,
            content=content,
        )

    def _log(self, message: str, run_id: str, session_id: str, level: int = logging.INFO) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"| {timestamp} | {session_id:<8} | {run_id:<8} | {message}"
        self.logger.log(level, message)
