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
    )


def main() -> None:
    agent = get_simple_agent()
    run_server(agent, MCP_CONFIG)


if __name__ == "__main__":
    main()
