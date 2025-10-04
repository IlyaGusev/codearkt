import os
import logging
from pathlib import Path

from phoenix.otel import register
import fire  # type: ignore

from codearkt.codeact import CodeActAgent
from codearkt.prompt_storage import PromptStorage
from codearkt.llm import LLM
from codearkt.server import run_server
from codearkt.otel import CodeActInstrumentor

current_dir = Path(__file__).parent

PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "codearkt")
ACADEMIA_MCP_URL = os.getenv("ACADEMIA_MCP_URL", "http://0.0.0.0:5056/mcp")
MCP_CONFIG = {"mcpServers": {"academia": {"url": ACADEMIA_MCP_URL, "transport": "streamable-http"}}}

LIBRARIAN_DESCRIPTION = """This team member runs gets and analyzes information from papers.
He has access to ArXiv, Semantic Scholar, ACL Anthology, and web search.
Ask him any questions about papers and web articles.
Give him your task as an only string argument. Follow the task format described above, include all the details."""


def get_librarian() -> CodeActAgent:
    llm = LLM(model_name="deepseek/deepseek-chat-v3-0324")
    prompts = PromptStorage.load(current_dir / "librarian.yaml")
    return CodeActAgent(
        name="librarian",
        description=LIBRARIAN_DESCRIPTION,
        llm=llm,
        prompts=prompts,
        tool_names=[
            "arxiv_download",
            "arxiv_search",
            "web_search",
            "visit_webpage",
        ],
        planning_interval=5,
        verbosity_level=logging.INFO,
    )


def get_manager(model_name: str) -> CodeActAgent:
    llm = LLM(model_name=model_name)
    return CodeActAgent(
        name="manager",
        description="A manager agent",
        llm=llm,
        managed_agents=[get_librarian()],
        tool_names=[],
        planning_interval=5,
        verbosity_level=logging.INFO,
    )


def main(port: int = 5055, model_name: str = "deepseek/deepseek-chat-v3-0324") -> None:
    register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=f"{PHOENIX_URL}/v1/traces",
        auto_instrument=True,
    )
    CodeActInstrumentor().instrument()
    agent = get_manager(model_name=model_name)
    run_server(agent, MCP_CONFIG, port=port)


if __name__ == "__main__":
    fire.Fire(main)
