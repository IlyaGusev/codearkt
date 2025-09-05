from typing import Any, Callable, Mapping, Tuple, Dict, Collection, Optional, List
from inspect import signature
import logging
from contextlib import suppress

from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    MessageAttributes,
)
from openinference.instrumentation.openai import OpenAIInstrumentor

from wrapt import wrap_function_wrapper  # type: ignore

from codearkt.codeact import CodeActAgent
from codearkt.python_executor import PythonExecutor, ExecResult
from codearkt.llm import ChatMessage

logger = logging.getLogger(__name__)

INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
SESSION_ID = SpanAttributes.SESSION_ID
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _get_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)
    return arguments


def _format_message_content(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    assert isinstance(message.content, list)
    content = ""
    if len(message.content) == 1 and message.content[0]["type"] == "text":
        content = message.content[0]["text"]
    else:
        content = str(message.content)
    return content


def _get_input_message_attributes(arguments: Dict[str, Any]) -> Dict[str, Any]:
    assert "messages" in arguments
    input_messages = arguments["messages"]
    message_attributes = {}
    for idx, message in enumerate(input_messages):
        message_attributes[f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}"] = message.role
        message_attributes[f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}"] = (
            _format_message_content(message)
        )
    return message_attributes


def _get_output_message_attributes(messages: List[ChatMessage]) -> Dict[str, Any]:
    message_attributes = {}
    for idx, message in enumerate(messages):
        message_attributes[f"{LLM_OUTPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}"] = message.role
        message_attributes[f"{LLM_OUTPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}"] = (
            _format_message_content(message)
        )
    return message_attributes


class _AinvokeWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        bound_args = _bind_arguments(wrapped, *args, **kwargs)
        session_id: Optional[str] = bound_args.get("session_id")
        span_name = f"Agent: {instance.name}"

        token: Optional[object] = None
        if session_id is not None:
            token = context_api.attach(context_api.set_value(SESSION_ID, session_id))

        try:
            arguments = _get_arguments(wrapped, *args, **kwargs)
            conversation = "\n\n".join(
                [f"{message.role}: {message.content}" for message in arguments["messages"]]
            )
            message_attributes = _get_input_message_attributes(arguments)

            with self._tracer.start_as_current_span(
                span_name,
                attributes={
                    OPENINFERENCE_SPAN_KIND: AGENT,
                    INPUT_VALUE: conversation,
                    **message_attributes,
                    **({SESSION_ID: session_id} if session_id is not None else {}),
                    **dict(get_attributes_from_context()),
                },
            ) as span:
                try:
                    result = await wrapped(*args, **kwargs)
                    span.set_status(trace_api.StatusCode.OK)
                    span.set_attribute(OUTPUT_VALUE, str(result))
                    return result
                except Exception as e:
                    with suppress(Exception):
                        span.record_exception(e)
                    span.set_status(trace_api.StatusCode.ERROR)
                    raise
        except Exception as e:
            raise e
        finally:
            if token is not None:
                context_api.detach(token)  # type: ignore[arg-type]


class _StepWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        arguments = _get_arguments(wrapped, *args, **kwargs)
        session_id: Optional[str] = arguments.get("session_id")
        step_number: Optional[int] = arguments.get("step_number")
        message_attributes = _get_input_message_attributes(arguments)
        input_messages = arguments["messages"]
        conversation = "\n\n".join(
            [f"{message.role}: {message.content}" for message in input_messages]
        )

        span_name = f"Step {step_number}"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
                INPUT_VALUE: conversation,
                **message_attributes,
                **({SESSION_ID: session_id} if session_id is not None else {}),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            try:
                result: List[ChatMessage] = await wrapped(*args, **kwargs)
                formatted_lines = []
                for message in result:
                    content = _format_message_content(message)
                    formatted_lines.append(f"{message.role}: {content}")
                conversation = "\n\n".join(formatted_lines)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, conversation)
                span.set_attributes(_get_output_message_attributes(result))
                return result
            except Exception as e:
                with suppress(Exception):
                    span.record_exception(e)
                span.set_status(trace_api.StatusCode.ERROR)
                raise


class _ToolWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_name = "PythonExecutor"

        arguments = _get_arguments(wrapped, *args, **kwargs)
        session_id: Optional[str] = getattr(instance, "session_id", None)

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: arguments["code"],
                **({SESSION_ID: session_id} if session_id is not None else {}),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            try:
                result: ExecResult = await wrapped(*args, **kwargs)
                message = result.to_message()
                content = _format_message_content(message)
            except Exception as e:
                with suppress(Exception):
                    span.record_exception(e)
                span.set_status(trace_api.StatusCode.ERROR)
                raise
            span.set_attribute(OUTPUT_VALUE, content)
            if getattr(result, "error", None):
                span.set_status(trace_api.StatusCode.ERROR)
            else:
                span.set_status(trace_api.StatusCode.OK)
            return result


class CodeActInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_step_method",
        "_original_ainvoke_method",
        "_original_final_method",
        "_original_tool_ainvoke_method",
        "_tracer",
    )

    _original_step_method: Optional[Callable[..., Any]]
    _original_ainvoke_method: Optional[Callable[..., Any]]
    _original_final_method: Optional[Callable[..., Any]]
    _original_tool_ainvoke_method: Optional[Callable[..., Any]]
    _tracer: OITracer

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["codearkt"]

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, "0.1.0", tracer_provider),
            config=config,
        )

        try:
            OpenAIInstrumentor().instrument(
                tracer_provider=tracer_provider,
                config=config,
            )
        except Exception as exc:
            logger.debug("Failed to instrument OpenAI: %s", exc)

        self._original_step_method = getattr(CodeActAgent, "_step", None)
        step_wrapper = _StepWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent._step",
            wrapper=step_wrapper,
        )

        self._original_final_method = getattr(CodeActAgent, "_handle_final_message", None)
        handle_final_message_wrapper = _StepWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent._handle_final_message",
            wrapper=handle_final_message_wrapper,
        )

        self._original_ainvoke_method = getattr(CodeActAgent, "ainvoke", None)
        ainvoke_wrapper = _AinvokeWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent.ainvoke",
            wrapper=ainvoke_wrapper,
        )

        self._original_tool_ainvoke_method = getattr(PythonExecutor, "ainvoke", None)
        tool_wrapper = _ToolWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.python_executor",
            name="PythonExecutor.ainvoke",
            wrapper=tool_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_step_method is not None:
            setattr(CodeActAgent, "_step", self._original_step_method)
            self._original_step_method = None
        if self._original_ainvoke_method is not None:
            setattr(CodeActAgent, "ainvoke", self._original_ainvoke_method)
            self._original_ainvoke_method = None
        if self._original_final_method is not None:
            setattr(CodeActAgent, "_handle_final_message", self._original_final_method)
            self._original_final_method = None
        if self._original_tool_ainvoke_method is not None:
            setattr(PythonExecutor, "ainvoke", self._original_tool_ainvoke_method)
            self._original_tool_ainvoke_method = None
