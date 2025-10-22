import asyncio
import uuid

from codearkt.util import get_unique_id

import httpx
from a2a.client import ClientConfig, minimal_agent_card
from a2a.client import ClientFactory
from a2a.types import Message, TextPart


async def send_simple_message() -> None:
    async with httpx.AsyncClient(timeout=86400) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client)
        factory = ClientFactory(config)
        card = minimal_agent_card("http://localhost:8000/a2a/agents/manager/")
        client = factory.create(card)

        context_id = get_unique_id()
        text_content = "Get name of the 2409.06820 paper"
        message = Message(
            messageId=str(uuid.uuid4()),
            parts=[TextPart(text=text_content)],
            role="user",
            contextId=context_id,
        )

        print("Sending message to agent...")
        print(f"User: {text_content}")
        print("\nAgent response:")
        print("-" * 60)

        async for event in client.send_message(message):
            task = event[0]
            artifacts = task.artifacts
            assert isinstance(artifacts, list)
            assert len(artifacts) > 0
            print(artifacts[0].parts[0].root.text)
        print("\n" + "-" * 60)

        text_content = "Name other papers from the same author"
        message = Message(
            messageId=str(uuid.uuid4()),
            parts=[TextPart(text=text_content)],
            role="user",
            contextId=context_id,
        )

        print("Sending message to agent...")
        print(f"User: {text_content}")
        print("\nAgent response:")
        print("-" * 60)

        async for event in client.send_message(message):
            task = event[0]
            artifacts = task.artifacts
            assert isinstance(artifacts, list)
            assert len(artifacts) > 0
            print(artifacts[0].parts[0].root.text)
        print("\n" + "-" * 60)


async def discover_agents() -> None:
    print("Discovering available agents...")
    print("=" * 60)

    async with httpx.AsyncClient() as http_client:
        response = await http_client.get("http://localhost:8000/a2a/agents")
        agents = response.json()

        print(f"\nFound {len(agents.get('agents', []))} agent(s):\n")

        for agent_info in agents.get("agents", []):
            print(f"  Name: {agent_info['name']}")
            print(f"  URL: {agent_info['url']}")
            print(f"  Agent Card: {agent_info['agent_card']}")

            # Fetch agent card
            card_response = await http_client.get(agent_info["agent_card"])
            card = card_response.json()

            print(f"  Description: {card.get('description', 'N/A')}")
            print(f"  Version: {card.get('version', 'N/A')}")

            capabilities = card.get("capabilities", {})
            print(f"  Streaming: {capabilities.get('streaming', False)}")

            skills = card.get("skills", [])
            if skills:
                print("  Skills:")
                for skill in skills:
                    print(f"    - {skill.get('name')}: {skill.get('description')}")

            print()

    print("=" * 60)


async def main() -> None:
    print("\n" + "=" * 60)
    print("CodeArkt A2A Client Examples")
    print("=" * 60 + "\n")

    await discover_agents()
    print("\n")
    await send_simple_message()

    print("\n\n")


if __name__ == "__main__":
    asyncio.run(main())
