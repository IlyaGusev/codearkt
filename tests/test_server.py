from httpx import ASGITransport, AsyncClient

from codearkt.server import get_main_app
from codearkt.codeact import CodeActAgent
from codearkt.llm import LLM
from codearkt.event_bus import AgentEventBus
from codearkt.settings import settings


async def test_agents_list_and_cancel_endpoint() -> None:
    agent = CodeActAgent(name="root", description="root", llm=LLM("gpt-4o"))
    app = get_main_app(
        agent,
        event_bus=AgentEventBus(),
        mcp_config=None,
        server_host=settings.DEFAULT_SERVER_HOST,
        server_port=0,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/agents/list")
        assert resp.status_code == 200
        cards = resp.json()
        assert any(card["name"] == "root" for card in cards)

        cancel = await client.post("/agents/cancel", json={"session_id": "nonexistent"})
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"
