import asyncio
import tempfile
from pathlib import Path
from textwrap import dedent
from datetime import datetime

from academia_mcp.tools import arxiv_search
from academia_mcp.tools import arxiv_download

from codearkt.codeact import CodeActAgent, extract_code_from_text
from codearkt.llm import ChatMessage, LLM
from codearkt.event_bus import AgentEventBus, EventType
from codearkt.util import get_unique_id
from codearkt.server import run_query, run_batch
from codearkt.metrics import TokenUsageStore

from tests.conftest import MCPServerTest, get_nested_agent


class TestExtractCodeFromText:
    def test_extract_code_from_text_basic(self) -> None:
        text = 'Code:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_line_breaks(self) -> None:
        text = 'Code:\n```python\n\n\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_py(self) -> None:
        text = 'Code:\n```py\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_change(self) -> None:
        text = 'Execute code:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_code_example(self) -> None:
        text = 'Code example:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code is None

    def test_extract_code_from_text_multiple(self) -> None:
        text = "Code:\n```py\na = 1\n```\nAnother code:\n```py\nb = 2\n```"
        code = extract_code_from_text(dedent(text))
        assert code == "a = 1\n\nb = 2", code

    def test_extract_code_from_text_code_unclosed(self) -> None:
        text = 'Code:\n```python\nprint("Hello, world!")\n'
        code = extract_code_from_text(dedent(text))
        assert code is None

    def test_extract_code_from_text_spaces_bug(self) -> None:
        text = 'Code:    \n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code is not None, code

    def test_extract_code_markdown(self) -> None:
        text = '**Code:**\n```py\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code is not None, code


class TestCodeActAgent:
    async def test_codeact_no_tools(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=[],
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
            session_id=get_unique_id(),
        )
        str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, result

    async def test_codeact_token_usage(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=[],
        )
        session_id = get_unique_id()
        token_usage_store = TokenUsageStore()
        await agent.ainvoke(
            [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
            session_id=session_id,
            token_usage_store=token_usage_store,
        )
        token_usage = token_usage_store.get(session_id)
        assert token_usage.prompt_tokens > 0, token_usage
        assert token_usage.completion_tokens > 0, token_usage

    async def test_codeact_gpt_5_mini(self, gpt_5_mini: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=gpt_5_mini,
            tool_names=[],
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
            session_id=get_unique_id(),
        )
        str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, result

    async def test_codeact_max_iterations(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            max_iterations=1,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
            server_host=mcp_server_test.host,
            server_port=mcp_server_test.port,
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_initial_plan(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            planning_interval=5,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
            server_host=mcp_server_test.host,
            server_port=mcp_server_test.port,
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_zero_iterations(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            max_iterations=0,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
            server_host=mcp_server_test.host,
            server_port=mcp_server_test.port,
        )
        assert "role-playing language models" not in str(result).lower(), result

    async def test_codeact_images(
        self, gpt_4o: LLM, mcp_server_test: MCPServerTest, test_image_url: str
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=gpt_4o,
            tool_names=["show_image"],
            max_iterations=5,
        )
        result = await agent.ainvoke(
            [
                ChatMessage(
                    role="user",
                    content=f"What blocks are in this image? {test_image_url}\nUse show_image tool",
                )
            ],
            session_id=get_unique_id(),
            server_host=mcp_server_test.host,
            server_port=mcp_server_test.port,
        )
        assert "Player" in str(result), result

    async def test_codeact_multi_agent(self, deepseek: LLM, mcp_server_test: MCPServerTest) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            managed_agents=[get_nested_agent()],
        )
        query = "Get the exact title of 2409.06820v4."
        result = await agent.ainvoke(
            [
                ChatMessage(
                    role="user",
                    content=query,
                )
            ],
            session_id=get_unique_id(),
            server_host=mcp_server_test.host,
            server_port=mcp_server_test.port,
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_event_bus_simple(self, deepseek: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
        )
        query = "What is 432412421249 * 4332144219?"
        event_bus = AgentEventBus()
        session_id = get_unique_id()
        task = asyncio.create_task(
            agent.ainvoke(
                [
                    ChatMessage(
                        role="user",
                        content=query,
                    )
                ],
                session_id=session_id,
                event_bus=event_bus,
            )
        )
        event_bus.register_task(
            session_id=session_id,
            agent_name=agent.name,
            task=task,
        )
        events = []
        async for event in event_bus.stream_events(session_id):
            events.append(event)

        assert len(events) > 0, events

        assert events[0].event_type == EventType.AGENT_START, events[0]
        assert events[0].agent_name == agent_name, events[0]
        assert events[0].session_id == session_id, events[0]
        assert events[0].content is None, events[0]

        assert events[-1].event_type == EventType.AGENT_END, events[-1]
        assert events[-1].agent_name == agent_name, events[-1]
        assert events[-1].session_id == session_id, events[-1]
        assert events[-1].content is None, events[-1]

        assert events[1].event_type == EventType.OUTPUT, events[1]
        assert events[1].agent_name == agent_name, events[1]
        assert events[1].session_id == session_id, events[1]
        assert events[1].content is not None, events[1]

        event_types = {event.event_type for event in events}
        assert EventType.TOOL_RESPONSE in event_types, event_types

        contents = [e.content for e in events if e.event_type == EventType.OUTPUT if e.content]
        final_text = "".join(contents)
        str_result = str(final_text).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, str_result

    async def test_run_query_simple(self, deepseek: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
        )
        result = await run_query("What is 432412421249 * 4332144219?", agent, {})
        str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, str_result

    async def test_run_query_additional_tools(self, deepseek: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search_1"],
        )
        result = await run_query(
            "Get the exact title of 2409.06820v4.",
            agent,
            {},
            additional_tools={"arxiv_search_1": arxiv_search},
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_run_batch(self, deepseek: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
        )
        with tempfile.NamedTemporaryFile(mode="w") as f:
            results = await run_batch(
                ["What is 432412421249 * 4332144219?", "Get the exact title of 2409.06820v4."] * 2,
                agent,
                additional_tools={"arxiv_search": arxiv_search},
                max_concurrency=4,
                output_path=Path(f.name),
            )
            text = Path(f.name).read_text()
            assert text is not None, text
            assert len(text) > 0, text
            assert len(text.strip().splitlines()) == 4, text
        assert len(results) == 4, results
        result1 = str(results[0]).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in result1, result1
        result2 = str(results[1]).lower()
        assert "role-playing language models" in result2, result2

    async def test_codeact_multi_agent_batch(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            managed_agents=[get_nested_agent()],
        )
        queries = ["Get the exact title of 2409.06820v4."] * 5
        results = await run_batch(
            queries,
            agent,
            max_concurrency=5,
            task_timeout=600,
            additional_tools={"arxiv_search": arxiv_search, "arxiv_download": arxiv_download},
        )
        assert len(results) == 5, results
        for result in results:
            assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_multi_agent_timeout_batch(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            managed_agents=[get_nested_agent()],
        )
        queries = ["Get the exact title of 2409.06820v4."] * 5
        results = await run_batch(
            queries,
            agent,
            max_concurrency=2,
            task_timeout=1,
            additional_tools={"arxiv_search": arxiv_search, "arxiv_download": arxiv_download},
        )
        assert len(results) == 5, results
        for result in results:
            assert "timeout" in str(result).lower(), result

    async def test_codeact_tool_description_dedent(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
        )
        tools = await agent._get_tools(
            server_host=mcp_server_test.host, server_port=mcp_server_test.port
        )
        assert tools[0].description is not None, tools[0].description
        assert tools[0].description.strip() == tools[0].description

    async def test_codeact_current_date(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="What is the current date? Use %Y-%m-%d format")],
            session_id=get_unique_id(),
        )
        current_date = datetime.now().strftime("%Y-%m-%d")
        assert current_date in str(result), result

    async def test_codeact_grok_code(self, grok_code: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=grok_code,
        )
        result = await run_query("What is 432412421249 * 4332144219?", agent, {})
        str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, str_result

    async def test_codeact_prompt_output_schema(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        host = mcp_server_test.host
        port = mcp_server_test.port
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
            tool_names=["structured_arxiv_download", "arxiv_search"],
        )
        tools = await agent._get_tools(server_host=host, server_port=port)
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = agent.prompts.system.render(tools=tools, current_date=current_date)
        assert "The title of the paper" in system_prompt
