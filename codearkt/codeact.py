import asyncio
import copy
import logging
import re
import traceback
from dataclasses import dataclass
from contextlib import suppress
from datetime import datetime
from textwrap import dedent
from typing import List, Optional, Self, Sequence

from mcp import Tool

from codearkt.event_bus import AgentEventBus, EventType
from codearkt.llm import LLM, ChatMessage, ChatMessages
from codearkt.metrics import TokenUsageStore
from codearkt.prompt_storage import PromptStorage
from codearkt.python_executor import PythonExecutor
from codearkt.settings import settings
from codearkt.tools import fetch_tools
from codearkt.util import get_unique_id, truncate_content


@dataclass
class RunContext:
    session_id: str
    run_id: str
    event_bus: Optional[AgentEventBus] = None
    token_usage_store: Optional[TokenUsageStore] = None


class CodeActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        tool_names: Sequence[str] = tuple(),
        prompts: Optional[PromptStorage] = None,
        max_iterations: int = settings.DEFAULT_MAX_ITERATIONS,
        verbosity_level: int = logging.ERROR,
        planning_interval: Optional[int] = None,
        managed_agents: Optional[List[Self]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.llm: LLM = llm
        self.prompts: PromptStorage = prompts or PromptStorage.default()
        self.tool_names = list(tool_names)
        self.max_iterations = max_iterations
        self.verbosity_level = verbosity_level
        self.planning_interval = planning_interval
        self.managed_agents: Optional[List[Self]] = managed_agents
        self.token_usage_store: TokenUsageStore = TokenUsageStore()

        if self.managed_agents:
            for agent in self.managed_agents:
                agent_tool_name = settings.AGENT_TOOL_PREFIX + agent.name
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
        token_usage_store: TokenUsageStore | None = None,
        server_host: str | None = None,
        server_port: int | None = None,
    ) -> str:
        messages = copy.deepcopy(messages)
        run_id = get_unique_id()
        run_context = RunContext(
            session_id=session_id,
            run_id=run_id,
            event_bus=event_bus,
            token_usage_store=token_usage_store,
        )
        await self._publish_event(run_context, event_type=EventType.AGENT_START)
        self._log(run_context, f"Starting agent {self.name}")

        python_executor = None
        try:
            python_executor = PythonExecutor(
                session_id=session_id,
                tool_names=self.tool_names,
                interpreter_id=run_id,
                tools_server_port=server_port,
                tools_server_host=server_host,
            )
            self._log(run_context, "Python interpreter started")
            self._log(run_context, f"Host: {server_host}, port: {server_port}")

            tools = await self._get_tools(server_host=server_host, server_port=server_port)
            self._log(run_context, f"Fetched tools: {[tool.name for tool in tools]}")

            current_date = datetime.now().strftime("%Y-%m-%d")
            system_prompt = self.prompts.system.render(tools=tools, current_date=current_date)

            if messages and messages[0].role not in ("system", "developer"):
                messages = [ChatMessage(role="system", content=system_prompt)] + messages

            for step_number in range(1, self.max_iterations + 1):
                if self.planning_interval and (step_number - 1) % self.planning_interval == 0:
                    self._log(run_context, f"Planning step {step_number} started")
                    new_messages = await self._run_planning_step(
                        messages=messages,
                        tools=tools,
                        run_context=run_context,
                    )
                    messages.extend(new_messages)
                    self._log(run_context, f"Planning step {step_number} completed")

                self._log(run_context, f"Step {step_number} started")
                new_messages = await self._step(
                    messages=messages,
                    python_executor=python_executor,
                    run_context=run_context,
                    step_number=step_number,
                )
                messages.extend(new_messages)
                self._log(run_context, f"Step {step_number} completed")
                if messages[-1].role == "assistant":
                    break
            else:
                new_messages = await self._handle_final_message(
                    messages=messages,
                    run_context=run_context,
                )
                messages.extend(new_messages)
                self._log(run_context, "Final step completed")

        except Exception as exc:
            error = traceback.format_exc()
            self._log_error(run_context, f"Agent {self.name} failed with error: {error}")
            raise exc

        finally:
            if python_executor:
                await python_executor.cleanup()
            await self._publish_event(run_context, EventType.AGENT_END)
            self._log(run_context, "Clean up completed")

        self._log(run_context, f"Agent {self.name} completed successfully")
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

    async def _run_llm(
        self,
        messages: ChatMessages,
        run_context: RunContext,
        excluded_stop_sequences: List[str],
        included_stop_sequences: List[str],
        event_type: EventType = EventType.OUTPUT,
    ) -> str:
        output_stream = self.llm.astream(messages=messages, stop=excluded_stop_sequences)

        output_text = ""
        last_usage = None
        try:
            async for event in output_stream:
                if event.usage:
                    last_usage = event.usage

                # Ignore everything after the stop sequence.
                # Can't just break because of the usage tracking.
                all_stop_sequences = excluded_stop_sequences + included_stop_sequences
                if any(ss in output_text for ss in all_stop_sequences):
                    continue

                delta = event.choices[0].delta
                if isinstance(delta.content, str):
                    chunk = delta.content
                elif isinstance(delta.content, list):
                    chunk = "\n".join([str(item) for item in delta.content])

                output_text += chunk
                await self._publish_event(run_context, event_type, chunk)
        finally:
            with suppress(Exception):
                await output_stream.aclose()

        for excluded_stop_sequence in excluded_stop_sequences:
            if excluded_stop_sequence in output_text:
                output_text = output_text.split(excluded_stop_sequence)[0]
                break

        for included_stop_sequence in included_stop_sequences:
            if included_stop_sequence in output_text:
                output_text = output_text.split(included_stop_sequence)[0]
                output_text += included_stop_sequence
                break

        await self._publish_event(run_context, event_type, "\n")
        if run_context.token_usage_store and last_usage:
            await run_context.token_usage_store.add(
                run_context.session_id,
                last_usage.prompt_tokens,
                last_usage.completion_tokens,
            )
        return output_text

    def _extract_code_from_text(self, text: str) -> str | None:
        begin_code_sequence = self.prompts.begin_code_sequence
        end_code_sequence = self.prompts.end_code_sequence
        pattern = rf"{begin_code_sequence}(.*?){end_code_sequence}"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(match.strip() for match in matches)
        return None

    def _extract_final_answer_from_text(self, content: str) -> str | None:
        begin_final_answer_sequence = self.prompts.begin_final_answer_sequence
        end_final_answer_sequence = self.prompts.end_final_answer_sequence
        pattern = rf"{begin_final_answer_sequence}(.*?){end_final_answer_sequence}"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return "\n\n".join(match.strip() for match in matches)
        return None

    async def _ensure_end_code_sequence(self, output_text: str, run_context: RunContext) -> str:
        begin_code = self.prompts.begin_code_sequence
        end_code = self.prompts.end_code_sequence
        if begin_code in output_text:
            last_begin = output_text.rfind(begin_code)
            text_after_begin = output_text[last_begin:]
            if end_code not in text_after_begin:
                chunk = end_code + "\n"
                await self._publish_event(run_context, EventType.OUTPUT, chunk)
                output_text += chunk
        return output_text

    async def _step(
        self,
        messages: ChatMessages,
        python_executor: PythonExecutor,
        run_context: RunContext,
        step_number: int,
    ) -> ChatMessages:
        self._log_debug(run_context, f"Step {step_number} inputs: {messages}")
        self._log(run_context, "LLM generates outputs...")
        output_text = ""
        try:
            output_text = await self._run_llm(
                messages=messages,
                run_context=run_context,
                excluded_stop_sequences=self.prompts.excluded_stop_sequences,
                included_stop_sequences=self.prompts.included_stop_sequences,
            )
            output_text = await self._ensure_end_code_sequence(
                output_text=output_text, run_context=run_context
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            exception = traceback.format_exc()
            error_text = f"LLM failed with error: {exception}. Please try again."
            self._log_error(run_context, error_text)
            return []

        self._log_debug(run_context, f"Step output: {output_text}")
        self._log(run_context, "LLM generated outputs!")

        code_action = self._extract_code_from_text(output_text)
        final_answer = self._extract_final_answer_from_text(output_text)
        self._log_debug(run_context, f"Code action: {code_action}")
        if final_answer is not None:
            self._log_debug(run_context, f"Final answer: {final_answer}")

        tool_call_message = ChatMessage(role="assistant", content=output_text)
        new_messages = [tool_call_message]

        if code_action is None and final_answer is None:
            self._log(run_context, "No tool calls or final answer detected")
            self._log_debug(run_context, f"Bad message: {output_text}")
            assert self.prompts.no_code_action is not None
            no_code_action_prompt = self.prompts.no_code_action.render()
            new_messages.append(ChatMessage(role="user", content=no_code_action_prompt))
            return new_messages

        elif code_action is None and final_answer is not None:
            self._log(run_context, "Final answer found!")
            self._log_debug(run_context, f"Final answer: {final_answer}")
            new_messages[-1].content = final_answer
            return new_messages

        assert code_action is not None
        # Final answer might be not None, but we ignore it.

        try:
            self._log(run_context, "Executing code...")
            code_result = await python_executor.ainvoke(code_action)
            self._log_debug(run_context, f"Code result: {code_result}")
            code_result_message: ChatMessage = code_result.to_message()
            assert isinstance(code_result_message.content, list)
            new_messages.append(code_result_message)
            tool_output: str = str(code_result_message.content[0]["text"]) + "\n"
            await self._publish_event(run_context, EventType.TOOL_RESPONSE, tool_output)
            self._log_debug(run_context, f"Tool output: {tool_output}")
            self._log(run_context, "Code was executed!")

        except asyncio.CancelledError:
            raise
        except Exception:
            exception = traceback.format_exc()
            new_messages.append(ChatMessage(role="user", content=f"Error: {exception}"))
            self._log_debug(run_context, f"Code error: {exception}")
            await self._publish_event(run_context, EventType.TOOL_RESPONSE, f"Error: {exception}\n")

        return new_messages

    async def _handle_final_message(
        self,
        messages: ChatMessages,
        run_context: RunContext,
    ) -> ChatMessages:
        prompt: str = self.prompts.final.render()
        final_message = ChatMessage(role="user", content=prompt)
        input_messages = messages + [final_message]
        self._log_debug(run_context, f"Final input messages: {input_messages}")

        output_text = await self._run_llm(
            messages=input_messages,
            run_context=run_context,
            excluded_stop_sequences=self.prompts.excluded_stop_sequences,
            included_stop_sequences=self.prompts.included_stop_sequences,
        )
        self._log_debug(run_context, f"Final message: {output_text}")

        return [ChatMessage(role="assistant", content=output_text)]

    def _process_messages_for_planning(
        self,
        messages: ChatMessages,
        last_n: int = settings.PLANNING_LAST_N,
        content_max_length: int = settings.PLANNING_CONTENT_MAX_LENGTH,
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
        run_context: RunContext,
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
            plan_prefix = self.prompts.plan_prefix.render().strip() + "\n\n"
            await self._publish_event(run_context, EventType.PLANNING_OUTPUT, plan_prefix)
            output_text = plan_prefix

            output_text += await self._run_llm(
                messages=input_messages,
                run_context=run_context,
                excluded_stop_sequences=[self.prompts.end_plan_sequence],
                included_stop_sequences=[self.prompts.end_plan_sequence],
                event_type=EventType.PLANNING_OUTPUT,
            )

            if self.prompts.end_plan_sequence in output_text:
                output_text = output_text.split(self.prompts.end_plan_sequence)[0].strip()

            # Always append the end plan sequence.
            output_text += self.prompts.end_plan_sequence

            plan_suffix = self.prompts.plan_suffix.render().strip()
            return [
                ChatMessage(role="assistant", content=output_text),
                ChatMessage(role="user", content=plan_suffix),
            ]

        except asyncio.CancelledError:
            raise
        except Exception:
            exception = traceback.format_exc()
            error_text = f"LLM failed with error: {exception}. Please try again."
            self._log_error(run_context, error_text)
            return []

    async def _publish_event(
        self,
        run_context: RunContext,
        event_type: EventType = EventType.OUTPUT,
        content: Optional[str] = None,
    ) -> None:
        if not run_context.event_bus:
            return
        await run_context.event_bus.publish_event(
            session_id=run_context.session_id,
            agent_name=self.name,
            event_type=event_type,
            content=content,
        )

    def _log(self, run_context: RunContext, message: str, level: int = logging.INFO) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = run_context.session_id
        run_id = run_context.run_id
        message = f"| {timestamp} | {session_id:<8} | {run_id:<8} | {message}"
        self.logger.log(level, message)

    def _log_debug(self, run_context: RunContext, message: str) -> None:
        self._log(run_context, message, level=logging.DEBUG)

    def _log_error(self, run_context: RunContext, message: str) -> None:
        self._log(run_context, message, level=logging.ERROR)
