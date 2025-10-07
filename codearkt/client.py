from typing import Iterator, List

import httpx
from pydantic import ValidationError

from codearkt.event_bus import AgentEvent
from codearkt.llm import ChatMessage
from codearkt.settings import settings

HEADERS = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}
REQUEST_TYPE = "POST"
CONNECT_TIMEOUT = 10.0


def _compose_base_url(host: str, port: int) -> str:
    base_url = f"{host}:{port}"
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    return base_url


def _compose_agent_url(host: str, port: int, agent_name: str) -> str:
    base_url = _compose_base_url(host, port)
    return f"{base_url}/agents/{agent_name}"


def _compose_cancel_url(host: str, port: int) -> str:
    base_url = _compose_base_url(host, port)
    return f"{base_url}/agents/cancel"


def query_agent(
    history: List[ChatMessage],
    *,
    session_id: str | None = None,
    host: str = settings.DEFAULT_SERVER_HOST,
    port: int = settings.DEFAULT_SERVER_PORT,
    agent_name: str = "manager",
) -> Iterator[AgentEvent]:
    url = _compose_agent_url(host, port, agent_name)

    serialized_history = [m.model_dump() for m in history]
    payload = {"messages": serialized_history, "stream": True}
    if session_id is not None:
        payload["session_id"] = session_id

    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, pool=None, read=None, write=None)
    with httpx.stream(
        REQUEST_TYPE, url, json=payload, headers=HEADERS, timeout=timeout
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if not chunk:
                continue
            try:
                yield AgentEvent.model_validate_json(chunk)
            except ValidationError:
                continue


def stop_agent(
    session_id: str,
    host: str = settings.DEFAULT_SERVER_HOST,
    port: int = settings.DEFAULT_SERVER_PORT,
) -> bool:
    url = _compose_cancel_url(host, port)
    payload = {"session_id": session_id}
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, pool=None, read=None, write=None)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False
