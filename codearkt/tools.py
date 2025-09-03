from typing import List
import httpx
import traceback

from mcp import ClientSession, Tool
from mcp.client.streamable_http import streamablehttp_client


async def fetch_tools(url: str) -> List[Tool]:
    all_tools: List[Tool] = []
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    try:
        async with streamablehttp_client(url + "/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                all_tools.extend(tools_response.tools)
    except Exception:
        traceback.print_exc()
        pass

    try:
        async with httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=0)) as client:
            resp = await client.get(url + "/agents/list")
            resp.raise_for_status()
            agent_cards = resp.json()
            for card in agent_cards:
                all_tools.append(
                    Tool(
                        name="agent__" + card["name"],
                        description=card["description"],
                        inputSchema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    )
                )
    except Exception:
        traceback.print_exc()
        pass

    return all_tools
