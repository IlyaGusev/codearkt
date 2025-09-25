import os
import copy
from typing import Dict, Any, List, cast, AsyncGenerator, Optional
from contextlib import suppress

from tiktoken import encoding_for_model
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HTTP_REFERRER = os.getenv("HTTP_REFERRER", "https://github.com/IlyaGusev/codearkt/")
X_TITLE = os.getenv("X_TITLE", "CodeArkt")
HEADERS = {
    "HTTP-Referer": HTTP_REFERRER,
    "X-Title": X_TITLE,
}


class FunctionCall(BaseModel):  # type: ignore
    name: str
    arguments: str


class ToolCall(BaseModel):  # type: ignore
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):  # type: ignore
    role: str
    content: str | List[Dict[str, Any]]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def __str__(self) -> str:
        dump: str = self.model_dump_json()
        return dump


ChatMessages = List[ChatMessage]
ChatStreamGenerator = AsyncGenerator[ChatCompletionChunk, None]


def count_openai_tokens(messages: ChatMessages) -> int:
    encoding = encoding_for_model("gpt-4o")
    tokens = [encoding.encode(str(message.content)) for message in messages]
    return sum(len(token) for token in tokens) + 3 * len(messages)


class LLM:
    def __init__(
        self,
        model_name: str,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        max_history_tokens: int = 200000,
        num_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._max_history_tokens = max_history_tokens
        self._params: Dict[str, Any] = {}
        self._num_retries = num_retries
        for k, v in kwargs.items():
            self._params[k] = v

    def _trim_messages(self, messages: ChatMessages) -> ChatMessages:
        if len(messages) <= 2:
            return messages
        tokens_count = count_openai_tokens(messages)
        first_messages = []
        while messages and messages[0].role in ("system", "developer"):
            first_messages.extend(messages[:1])
            messages = messages[1:]
        if messages[0].role == "user":
            first_messages.extend(messages[:1])
            messages = messages[1:]
        while tokens_count > self._max_history_tokens and len(messages) >= 2:
            tokens_count -= count_openai_tokens(messages[:2])
            messages = messages[2:]
        if first_messages:
            messages = first_messages + messages
        return messages

    async def astream(self, messages: ChatMessages, **kwargs: Any) -> ChatStreamGenerator:
        messages = copy.deepcopy(messages)
        api_params = {**self._params, **kwargs}

        if "gpt-5" in self._model_name:
            if messages[0].role == "system":
                messages[0].role = "developer"
            api_params.pop("stop", None)
            if "max_tokens" in api_params:
                api_params["max_completion_tokens"] = api_params.pop("max_tokens")

        if "grok-code" in self._model_name:
            api_params.pop("stop", None)

        messages = self._trim_messages(messages)
        casted_messages = [
            cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))
            for message in messages
        ]

        async with AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._num_retries,
        ) as api:
            stream: AsyncStream[ChatCompletionChunk] = await api.chat.completions.create(
                model=self._model_name,
                messages=casted_messages,
                stream=True,
                extra_headers=HEADERS,
                **api_params,
            )
            try:
                async for event in stream:
                    event_typed: ChatCompletionChunk = event
                    yield event_typed
            finally:
                with suppress(Exception):
                    await stream.close()
