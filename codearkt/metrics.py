import asyncio
from typing import Dict

from pydantic import BaseModel


class TokenUsage(BaseModel):  # type: ignore
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenUsageStore:
    def __init__(self) -> None:
        self._data: Dict[str, TokenUsage] = {}
        self._lock = asyncio.Lock()

    async def add(
        self, session_id: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        async with self._lock:
            usage = self._data.get(session_id, TokenUsage())
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            self._data[session_id] = usage

    async def pop(self, session_id: str) -> TokenUsage:
        async with self._lock:
            return self._data.pop(session_id, TokenUsage())

    def get(self, session_id: str) -> TokenUsage:
        return self._data.get(session_id, TokenUsage())
