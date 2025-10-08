from codearkt.client import (
    _compose_agent_url,
    _compose_base_url,
    _compose_cancel_url,
    query_agent,
    stop_agent,
)
from codearkt.event_bus import EventType
from codearkt.llm import ChatMessage
from codearkt.util import get_unique_id
from tests.conftest import MCPServerTest


class TestUrlComposition:
    def test_client_url_compose_base_url_with_http(self) -> None:
        url = _compose_base_url("http://localhost", 8000)
        assert url == "http://localhost:8000"

    def test_client_url_compose_base_url_without_http(self) -> None:
        url = _compose_base_url("localhost", 8000)
        assert url == "http://localhost:8000"

    def test_client_url_compose_base_url_with_https(self) -> None:
        url = _compose_base_url("https://example.com", 443)
        assert url == "https://example.com:443"

    def test_client_url_compose_agent_url(self) -> None:
        url = _compose_agent_url("localhost", 8000, "manager")
        assert url == "http://localhost:8000/agents/manager"

    def test_client_url_compose_agent_url_custom_agent(self) -> None:
        url = _compose_agent_url("http://example.com", 9000, "custom_agent")
        assert url == "http://example.com:9000/agents/custom_agent"

    def test_client_url_compose_cancel_url(self) -> None:
        url = _compose_cancel_url("localhost", 8000)
        assert url == "http://localhost:8000/agents/cancel"

    def test_client_url_compose_cancel_url_with_https(self) -> None:
        url = _compose_cancel_url("https://example.com", 443)
        assert url == "https://example.com:443/agents/cancel"


async def test_client_base(mcp_server_test: MCPServerTest) -> None:
    history = [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")]
    output_text = ""
    session_id = get_unique_id()
    for event in query_agent(
        history,
        host=mcp_server_test.host,
        port=mcp_server_test.port,
        agent_name="nested_agent",
        session_id=session_id,
    ):
        if event.event_type == EventType.OUTPUT and event.content:
            output_text += event.content
    output_text = output_text.replace(",", "").replace(".", "")
    assert "1873272970937648109531" in output_text, output_text


async def test_client_stop(mcp_server_test: MCPServerTest) -> None:
    history = [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")]
    output_text = ""
    session_id = get_unique_id()
    for event in query_agent(
        history,
        host=mcp_server_test.host,
        port=mcp_server_test.port,
        agent_name="nested_agent",
        session_id=session_id,
    ):
        if event.event_type == EventType.OUTPUT and event.content:
            output_text += event.content
        stop_agent(session_id, host=mcp_server_test.host, port=mcp_server_test.port)
    assert "1873272970937648109531" not in output_text
