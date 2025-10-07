import asyncio
import atexit
import textwrap
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

from docker import from_env as docker_from_env
from docker.client import DockerClient
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.networks import Network
from httpx import AsyncClient, HTTPError, Limits, RequestError, TimeoutException
from pydantic import BaseModel, ValidationError

from codearkt.llm import ChatMessage
from codearkt.settings import settings
from codearkt.tools import fetch_tools
from codearkt.util import get_unique_id, is_correct_json, truncate_content

IMAGE: str = settings.EXECUTOR_IMAGE
EXTERNAL_URL_ENV = settings.CODEARKT_EXECUTOR_URL

_CLIENT: Optional[DockerClient] = None
_CONTAINER: Optional[Container] = None
_DOCKER_LOCK: threading.Lock = threading.Lock()


def _cleanup_container() -> None:
    global _CONTAINER

    acquired: bool = _DOCKER_LOCK.acquire(timeout=settings.DOCKER_CLEANUP_TIMEOUT)
    try:
        if acquired and _CONTAINER:
            try:
                _CONTAINER.stop(timeout=settings.DOCKER_CLEANUP_TIMEOUT)
                _CONTAINER.remove(force=True)
                _CONTAINER = None
            except DockerException:
                pass
    finally:
        if acquired:
            _DOCKER_LOCK.release()


atexit.register(_cleanup_container)


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
                full_url = f"data:image/png;base64,{image_base64}"
                image_content = [{"type": "image_url", "image_url": {"url": full_url}}]
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


def _init_docker() -> DockerClient:
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


def _run_network(client: DockerClient) -> Network:
    name = settings.DOCKER_NET_NAME
    try:
        net = client.networks.get(name)
    except NotFound:
        net = client.networks.create(name, driver="bridge")
    return net


def _run_container(client: DockerClient, net_name: str) -> Container:
    return client.containers.run(
        IMAGE,
        detach=True,
        auto_remove=True,
        ports={"8000/tcp": None},
        mem_limit=settings.DOCKER_MEM_LIMIT,
        cpu_period=settings.DOCKER_CPU_PERIOD,
        cpu_quota=settings.DOCKER_CPU_QUOTA,
        pids_limit=settings.DOCKER_PIDS_LIMIT,
        cap_drop=["ALL"],
        read_only=True,
        tmpfs={"/tmp": "rw,size=64m", "/run": "rw,size=16m"},
        security_opt=["no-new-privileges"],
        extra_hosts={"host.docker.internal": "host-gateway"},
        sysctls={"net.ipv4.ip_forward": "0"},
        network=net_name,
        dns=[],
    )


def _get_url_from_container(container: Container) -> str:
    container.reload()
    ports = container.attrs["NetworkSettings"]["Ports"]
    mapping = ports["8000/tcp"][0]
    return f"http://localhost:{mapping['HostPort']}"


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
        self.is_ready = False

        if EXTERNAL_URL_ENV:
            self.url = EXTERNAL_URL_ENV.rstrip("/")
            return

        global _CLIENT, _CONTAINER
        with _DOCKER_LOCK:
            if not _CLIENT:
                _CLIENT = _init_docker()
            client = _CLIENT
            if not _CONTAINER:
                net = _run_network(client)
                _CONTAINER = _run_container(client, str(net.name))
        self.url = _get_url_from_container(_CONTAINER)

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
                resp = await client.post(
                    f"{self.url}/exec", json=payload, timeout=settings.EXEC_TIMEOUT
                )
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

    async def cleanup(self) -> None:
        payload = {"interpreter_id": self.interpreter_id}
        try:
            async with AsyncClient(limits=Limits(keepalive_expiry=0)) as client:
                response = await client.post(
                    f"{self.url}/cleanup", json=payload, timeout=settings.DOCKER_CLEANUP_TIMEOUT
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
