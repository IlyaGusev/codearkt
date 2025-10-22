import logging

from codearkt.server import run_server
from codearkt.codeact import CodeActAgent
from codearkt.llm import LLM

MCP_CONFIG = {
    "mcpServers": {"academia": {"url": "http://0.0.0.0:5056/mcp", "transport": "streamable-http"}}
}


def get_simple_agent() -> CodeActAgent:
    return CodeActAgent(
        name="manager",
        description="A simple agent",
        llm=LLM(model_name="deepseek/deepseek-chat-v3-0324"),
        tool_names=["arxiv_download", "arxiv_search"],
        verbosity_level=logging.INFO,
    )


def main() -> None:
    agent = get_simple_agent()
    mcp_config = MCP_CONFIG

    print("Starting CodeArkt A2A Server...")
    print("=" * 60)
    print("  List Agents: http://localhost:8000/a2a/agents")
    print("\n" + "=" * 60)

    run_server(
        agent=agent,
        mcp_config=mcp_config,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
