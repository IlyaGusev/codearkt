import asyncio
import threading
import time
import logging
import base64
from io import BytesIO
from typing import Generator, Dict
from contextlib import suppress

import httpx
import pytest
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from academia_mcp.tools import arxiv_download, arxiv_search, document_qa

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


def show_image(url: str) -> Dict[str, str]:
    """
    Reads an image from the specified URL.
    Always call this function at the end of the code block.
    For instance:
    ```python
    show_image("https://example.com/image.png")
    ```
    Do not print it ever, just return as the last expression.

    Returns an dictionary with a single "image" key.
    Args:
        url: Path to file or directory inside current work directory or web URL. Should not be absolute.
    """
    assert url.startswith("http")
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    buffer_io = BytesIO()
    image.save(buffer_io, format="PNG")
    img_bytes = buffer_io.getvalue()
    return {"image_base64": base64.b64encode(img_bytes).decode("utf-8")}


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
