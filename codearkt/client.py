from typing import Iterator, List

import httpx
from pydantic import ValidationError

from codearkt.event_bus import AgentEvent
from codearkt.llm import ChatMessage
from codearkt.server import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT

HEADERS = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}


def query_agent(
    history: List[ChatMessage],
    *,
    session_id: str | None = None,
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    agent_name: str = "manager",
) -> Iterator[AgentEvent]:
    base_url = f"{host}:{port}"
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    url = f"{base_url}/agents/{agent_name}"
    serialized_history = [m.model_dump() for m in history]
    payload = {"messages": serialized_history, "stream": True}
    if session_id is not None:
        payload["session_id"] = session_id

    timeout = httpx.Timeout(connect=10, pool=None, read=None, write=None)
    with httpx.stream("POST", url, json=payload, headers=HEADERS, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if not chunk:
                continue
            try:
                yield AgentEvent.model_validate_json(chunk)
            except ValidationError:
                continue


def stop_agent(
    session_id: str, host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT
) -> bool:
    base_url = f"{host}:{port}"
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    try:
        httpx.post(f"{base_url}/agents/cancel", json={"session_id": session_id}, timeout=5.0)
        return True
    except httpx.HTTPError:
        return False
