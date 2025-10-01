import asyncio
import json
import threading
import time
import logging
from typing import Generator, List, Optional
from contextlib import suppress

import pytest
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from academia_mcp.tools import arxiv_download, arxiv_search, document_qa, show_image

from codearkt.llm import LLM
from codearkt.codeact import CodeActAgent
from codearkt.server import get_agent_app, DEFAULT_SERVER_HOST, reset_app_status
from codearkt.event_bus import AgentEventBus

load_dotenv()
for name in ("httpx", "mcp", "openai", "uvicorn"):
    logging.getLogger(name).setLevel(logging.WARNING)


@pytest.fixture
def gpt_4o() -> LLM:
    return LLM(model_name="gpt-4o")


@pytest.fixture
def deepseek() -> LLM:
    return LLM(model_name="deepseek/deepseek-chat-v3-0324")


@pytest.fixture
def deepseek_small_context() -> LLM:
    return LLM(model_name="deepseek/deepseek-chat-v3-0324", max_history_tokens=100)


@pytest.fixture
def grok_code() -> LLM:
    return LLM(model_name="x-ai/grok-code-fast-1")


@pytest.fixture
def gpt_5_mini() -> LLM:
    return LLM(model_name="gpt-5-mini", tool_choice="auto")


@pytest.fixture
def test_image_url() -> str:
    return "https://arxiv.org/html/2409.06820v4/extracted/6347978/pingpong_v3.drawio.png"


def get_nested_agent(verbosity_level: int = logging.ERROR) -> CodeActAgent:
    return CodeActAgent(
        name="nested_agent",
        description="Call it when you need to get info about papers. Pass only your query as an argument.",
        llm=LLM(model_name="gpt-4o"),
        tool_names=("arxiv_download", "arxiv_search"),
        verbosity_level=verbosity_level,
    )


class NestedModel(BaseModel):  # type: ignore
    field: str = Field(description="The field of the nested model")


class StructuredDownloadResult(BaseModel):  # type: ignore
    title: str = Field(description="The title of the paper")
    abstract: str = Field(description="The abstract of the paper")
    toc: str | None = Field(description="The table of contents of the paper", default=None)
    sections: List[str] | None = Field(description="The sections of the paper", default=None)
    citations: List[str] | None = Field(description="The citations of the paper", default=None)
    nested_model: Optional[List[NestedModel]] = Field(
        description="The nested model for test", default=None
    )


def structured_arxiv_download(paper_id: str) -> StructuredDownloadResult:
    """
    Download a paper from arXiv.
    Args:
        paper_id: The ID of the paper to download.
    """
    result = json.loads(arxiv_download(paper_id=paper_id))
    deserialized: StructuredDownloadResult = StructuredDownloadResult.model_validate(result)
    return deserialized


class MCPServerTest:
    def __init__(self, port: int, host: str = DEFAULT_SERVER_HOST) -> None:
        self.port = port
        self.host = host
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

        reset_app_status()
        event_bus = AgentEventBus()
        mcp_server = FastMCP("Academia MCP", stateless_http=True)
        mcp_server.add_tool(arxiv_search)
        mcp_server.add_tool(arxiv_download)
        mcp_server.add_tool(document_qa)
        mcp_server.add_tool(show_image)
        mcp_server.add_tool(structured_arxiv_download, structured_output=True)
        app = mcp_server.streamable_http_app()
        agent_app = get_agent_app(
            get_nested_agent(),
            server_host=host,
            server_port=self.port,
            event_bus=event_bus,
        )
        app.mount("/agents", agent_app)
        config = uvicorn.Config(
            app,
            host=host,
            port=self.port,
            log_level="error",
            access_log=False,
            lifespan="on",
            ws="none",
        )
        self.server: uvicorn.Server = uvicorn.Server(config)

    def start(self) -> None:
        def _run() -> None:
            async def _serve() -> None:
                assert self.server is not None
                await self.server.serve()

            with suppress(asyncio.CancelledError):
                asyncio.run(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        deadline = time.time() + 30
        while time.time() < deadline:
            if self.server.started:
                self._started.set()
                break
            time.sleep(0.05)
        if not self._started.is_set():
            raise RuntimeError("Mock MCP server failed to start within 30 s")

    def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        reset_app_status()
        self.app = None

    def is_running(self) -> bool:
        return self._started.is_set() and self._thread is not None and self._thread.is_alive()


@pytest.fixture(scope="function")
def mcp_server_test() -> Generator[MCPServerTest, None, None]:
    server = MCPServerTest(port=6000)
    server.start()
    yield server
    server.stop()
