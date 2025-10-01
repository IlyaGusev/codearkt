import textwrap
import time
import os
import asyncio
import atexit
import traceback
from typing import Optional, Any, List, Dict, Sequence
import threading

from docker import from_env as docker_from_env
from docker.models.containers import Container
from docker.client import DockerClient
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.networks import Network
from pydantic import BaseModel, ValidationError
from httpx import AsyncClient, HTTPError, TimeoutException, Limits, RequestError

from codearkt.llm import ChatMessage
from codearkt.tools import fetch_tools
from codearkt.util import get_unique_id, truncate_content, is_correct_json


SHA_DIGEST: str = "sha256:8866223c75d644f51d0da5f12a8400ae4aa62262d5331b0e69dc7a3576055873"
DEFAULT_IMAGE: str = f"phoenix120/codearkt_http@{SHA_DIGEST}"
IMAGE: str = os.getenv("CODEARKT_EXECUTOR_IMAGE", DEFAULT_IMAGE)
MEM_LIMIT: str = "1g"
CPU_QUOTA: int = 50000
CPU_PERIOD: int = 100000
EXEC_TIMEOUT: int = 24 * 60 * 60  # 24 hours
CLEANUP_TIMEOUT: int = 10
PIDS_LIMIT: int = 256
NET_NAME: str = "codearkt_sandbox_net"
EXTERNAL_URL_ENV = os.getenv("CODEARKT_EXECUTOR_URL")

_CLIENT: Optional[DockerClient] = None
_CONTAINER: Optional[Container] = None
_DOCKER_LOCK: threading.Lock = threading.Lock()


def cleanup_container() -> None:
    global _CONTAINER

    acquired: bool = _DOCKER_LOCK.acquire(timeout=CLEANUP_TIMEOUT)
    try:
        if acquired and _CONTAINER:
            try:
                _CONTAINER.stop(timeout=CLEANUP_TIMEOUT)
                _CONTAINER.remove(force=True)
                _CONTAINER = None
            except DockerException:
                pass
    finally:
        if acquired:
            _DOCKER_LOCK.release()


atexit.register(cleanup_container)


class ExecResult(BaseModel):  # type: ignore
    stdout: str
    error: str | None = None
    result: Any | None = None

    def to_message(self) -> ChatMessage:
        image_content: List[Dict[str, Any]] | None = None
        output: str = ""
        if self.stdout:
            output += "Output:\n" + self.stdout + "\n\n"

        if self.result:
            if isinstance(self.result, dict) and "image_base64" in self.result:
                image_base64 = self.result["image_base64"]
                image_content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    }
                ]
            if not image_content:
                output += "Last expression:\n" + str(self.result) + "\n\n"

        if self.error:
            output += "Error: " + self.error

        output = output.strip()

        content = []
        if output:
            content.append({"type": "text", "text": output})
        if image_content:
            content += image_content
        return ChatMessage(role="user", content=content)


def init_docker() -> DockerClient:
    client = docker_from_env()
    try:
        client.ping()  # type: ignore
    except DockerException as exc:
        raise RuntimeError(
            "Docker daemon is not running or not accessible – skipping PythonExecutor setup."
        ) from exc

    try:
        client.images.get(IMAGE)
    except ImageNotFound:
        try:
            client.images.pull(IMAGE)
        except DockerException as exc:
            raise RuntimeError(
                f"Docker image '{IMAGE}' not found locally and failed to pull automatically."
            ) from exc
    except DockerException as exc:
        raise RuntimeError("Failed to query Docker images – ensure Docker is available.") from exc
    return client


def run_network(client: DockerClient) -> Network:
    try:
        net = client.networks.get(NET_NAME)
    except NotFound:
        net = client.networks.create(
            NET_NAME,
            driver="bridge",
        )
    return net


def run_container(client: DockerClient, net_name: str) -> Container:
    return client.containers.run(
        IMAGE,
        detach=True,
        auto_remove=True,
        ports={"8000/tcp": None},
        mem_limit=MEM_LIMIT,
        cpu_period=CPU_PERIOD,
        cpu_quota=CPU_QUOTA,
        pids_limit=PIDS_LIMIT,
        cap_drop=["ALL"],
        read_only=True,
        tmpfs={"/tmp": "rw,size=64m", "/run": "rw,size=16m"},
        security_opt=["no-new-privileges"],
        extra_hosts={"host.docker.internal": "host-gateway"},
        sysctls={"net.ipv4.ip_forward": "0"},
        network=net_name,
        dns=[],
    )


class PythonExecutor:
    def __init__(
        self,
        tool_names: Sequence[str] = tuple(),
        session_id: Optional[str] = None,
        tools_server_host: Optional[str] = None,
        tools_server_port: Optional[int] = None,
        interpreter_id: Optional[str] = None,
    ) -> None:
        self.tools_server_host = tools_server_host
        self.tools_server_port = tools_server_port
        self.session_id = session_id
        self.interpreter_id: str = interpreter_id or get_unique_id()
        self.tool_names = tool_names
        self.tools_are_checked = False

        global _CLIENT, _CONTAINER
        self.container = None
        if EXTERNAL_URL_ENV:
            self.url = EXTERNAL_URL_ENV.rstrip("/")
        else:
            with _DOCKER_LOCK:
                if not _CLIENT:
                    _CLIENT = init_docker()
                client = _CLIENT

                if not _CONTAINER:
                    net = run_network(client)
                    _CONTAINER = run_container(client, str(net.name))

                self.container = _CONTAINER
            self.url = self._get_url()
        self.is_ready = False

    async def ainvoke(self, code: str) -> ExecResult:
        if not self.tools_are_checked:
            await self._check_tools()
            self.tools_are_checked = True

        if not self.is_ready:
            await self._wait_for_ready()
            self.is_ready = True

        result = await self._call_exec(code)
        return result

    def _are_tools_available(self) -> bool:
        return bool(self.tool_names and self.tools_server_host and self.tools_server_port)

    async def _check_tools(self) -> None:
        assert not self.tools_are_checked

        available_tool_names = []
        if self._are_tools_available():
            server_url = f"{self.tools_server_host}:{self.tools_server_port}"
            available_tools = await fetch_tools(server_url)
            available_tool_names = [tool.name for tool in available_tools]

        for tool_name in self.tool_names:
            if tool_name.startswith("agent__"):
                continue
            if tool_name not in available_tool_names:
                raise ValueError(f"Tool {tool_name} not found in {available_tool_names}")

    async def _call_exec(self, code: str, send_tools: bool = True) -> ExecResult:
        payload = {
            "code": textwrap.dedent(code),
            "session_id": self.session_id,
            "tool_server_port": self.tools_server_port,
            "tool_names": self.tool_names if send_tools and self._are_tools_available() else [],
            "interpreter_id": self.interpreter_id,
        }

        try:
            async with AsyncClient(limits=Limits(keepalive_expiry=0)) as client:
                resp = await client.post(f"{self.url}/exec", json=payload, timeout=EXEC_TIMEOUT)
                resp.raise_for_status()
                out = resp.json()
                result: ExecResult = ExecResult.model_validate(out)
        except (HTTPError, TimeoutException, ValueError, ValidationError):
            result = ExecResult(stdout="", error=traceback.format_exc())

        if result.stdout:
            result.stdout = truncate_content(result.stdout)
        if result.error:
            result.error = truncate_content(result.error)
        if result.result and isinstance(result.result, str):
            if not is_correct_json(result.result):
                result.result = truncate_content(result.result)
        return result

    def _get_url(self) -> str:
        assert self.container is not None
        self.container.reload()
        ports = self.container.attrs["NetworkSettings"]["Ports"]
        mapping = ports["8000/tcp"][0]
        return f"http://localhost:{mapping['HostPort']}"

    async def cleanup(self) -> None:
        payload = {
            "interpreter_id": self.interpreter_id,
        }
        try:
            async with AsyncClient(limits=Limits(keepalive_expiry=0)) as client:
                response = await client.post(
                    f"{self.url}/cleanup", json=payload, timeout=CLEANUP_TIMEOUT
                )
                response.raise_for_status()
        except (HTTPError, TimeoutException):
            pass

    async def _wait_for_ready(self, max_wait: int = 60) -> None:
        delay = 0.1
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                output = await self._call_exec("print('ready')", send_tools=False)
                if output.stdout.strip() == "ready":
                    return
            except (RequestError, TimeoutException, AssertionError):
                pass

            await asyncio.sleep(delay)
            delay = min(delay * 2, 3.0)
        raise RuntimeError("Container failed to become ready within timeout")
