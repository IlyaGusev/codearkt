import ast
import asyncio
import contextlib
import io
import linecache
import logging
import multiprocessing
import time
import traceback
from collections import defaultdict
from contextlib import suppress
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel
from tools import fetch_tools  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="CodeArkt code runtime")
WORKERS: Dict[str, "Worker"] = {}
INTERPRETER_LOCKS: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@app.middleware("http")  # type: ignore
async def log_requests(request: Request, call_next: Callable[[Request], Any]) -> Any:
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logging.info(
        "%s %s completed in %.2f ms with status %d",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    return response


class Payload(BaseModel):  # type: ignore
    code: str
    tool_names: List[str]
    tool_server_port: Optional[int] = None
    interpreter_id: Optional[str] = None
    session_id: Optional[str] = None


class CleanupPayload(BaseModel):  # type: ignore
    interpreter_id: str


class ExecResult(BaseModel):  # type: ignore
    stdout: str
    error: str
    result: Any | None = None


def _exec_with_return(
    code: str,
    globals: Dict[str, Any],
    locals: Dict[str, Any] | None = None,
    filename: str = "snippet.py",
) -> Any:
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)

    mod = ast.parse(code, filename=filename, mode="exec")

    last_expr_node = None
    last_assigned_name: str | None = None
    if mod.body:
        last = mod.body[-1]
        if isinstance(last, ast.Expr):
            last_expr_node = last.value
            mod.body.pop()
        elif isinstance(last, ast.Assign):
            if last.targets and isinstance(last.targets[0], ast.Name):
                last_assigned_name = last.targets[0].id
        elif isinstance(last, ast.AnnAssign):
            if last.value is not None and isinstance(last.target, ast.Name):
                last_assigned_name = last.target.id
        elif isinstance(last, ast.AugAssign):
            if isinstance(last.target, ast.Name):
                last_assigned_name = last.target.id

    exec_code = compile(mod, filename, "exec")
    globals.setdefault("__name__", "__main__")
    globals.setdefault("__file__", filename)
    exec(exec_code, globals, locals)

    if last_expr_node is not None:
        expr_ast = ast.Expression(last_expr_node)
        ast.fix_missing_locations(expr_ast)
        eval_code = compile(expr_ast, filename, "eval")
        return eval(eval_code, globals, locals)

    if last_assigned_name is not None:
        ns = locals if locals is not None else globals
        return ns.get(last_assigned_name)

    return None


def _execute_code(
    code: str,
    globals_dict: Dict[str, Any],
    filename: str = "snippet.py",
) -> ExecResult:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = _exec_with_return(code, globals_dict, filename=filename)
        return ExecResult(stdout=buf.getvalue(), error="", result=result)
    except SyntaxError as e:
        line = (
            e.text
            if e.text is not None
            else linecache.getline(e.filename or filename, e.lineno or 0)
        )
        caret = ""
        if e.offset and e.offset > 0:
            caret = " " * (e.offset - 1) + "^"
        msg = f"{e.filename}:{e.lineno}:{e.offset}: {e.msg}\n{line}{caret}"
        return ExecResult(stdout=buf.getvalue(), error=msg, result=None)
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        filtered_frames = [f for f in tb.stack if f.filename == filename]
        if filtered_frames:
            tb.stack = type(tb.stack).from_list(filtered_frames)
        return ExecResult(stdout=buf.getvalue(), error="".join(tb.format()), result=None)


def _worker_main(request_q: multiprocessing.Queue, response_q: multiprocessing.Queue) -> None:  # type: ignore
    import asyncio

    _tools: Dict[str, Any] = {}
    _globals: Dict[str, Any] = {"__name__": "__main__"}

    while True:
        item = request_q.get()
        if item is None:
            break
        code: str = item["code"]
        tool_server_port: int = item["tool_server_port"]
        tool_names: List[str] = item["tool_names"]
        session_id: Optional[str] = item["session_id"]
        current_tools: Dict[str, Any] = dict()

        if tool_names:
            try:
                if not _tools or any(name not in _tools for name in tool_names):
                    _tools = asyncio.run(fetch_tools(tool_server_port))

                current_tools = {name: _tools[name] for name in tool_names}
                for name, fn in current_tools.items():
                    if session_id and name.startswith("agent__"):
                        _globals[name] = partial(fn, session_id=session_id)
                    else:
                        _globals[name] = fn
            except Exception:
                print("Error fetching tools", traceback.format_exc())
                exec_result = ExecResult(stdout="", error=traceback.format_exc(), result=None)
                response_q.put(exec_result.model_dump())
                continue

        # Remove tools that are no longer requested
        unused = set(_globals.keys()) & set(_tools.keys()) - set(current_tools.keys())
        for name in unused:
            _globals.pop(name, None)

        exec_result = _execute_code(code, _globals)
        response_q.put(exec_result.model_dump())


class Worker:
    def __init__(self) -> None:
        self._request_q: multiprocessing.Queue[Optional[Dict[str, Any]]] = multiprocessing.Queue()
        self._response_q: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_worker_main,
            args=(self._request_q, self._response_q),
            daemon=True,
        )
        self._process.start()

    def exec(
        self,
        code: str,
        tool_server_port: Optional[int],
        tool_names: List[str],
        session_id: Optional[str],
    ) -> Any:
        self._request_q.put(
            {
                "code": code,
                "tool_server_port": tool_server_port,
                "tool_names": tool_names,
                "session_id": session_id,
            }
        )
        return self._response_q.get()

    def terminate(self) -> None:
        self._request_q.put(None)
        if self._process.is_alive():
            with suppress(Exception):
                self._process.terminate()
            self._process.join(timeout=2)

        if self._process.is_alive():
            with suppress(Exception):
                self._process.kill()
            self._process.join(timeout=1)

        with suppress(Exception):
            self._request_q.close()
        with suppress(Exception):
            self._response_q.close()


def _get_worker(interpreter_id: str) -> Worker:
    if interpreter_id not in WORKERS:
        WORKERS[interpreter_id] = Worker()
    return WORKERS[interpreter_id]


@app.post("/exec")  # type: ignore
async def exec_code(payload: Payload) -> ExecResult:
    interpreter_id = payload.interpreter_id or "default"
    lock = INTERPRETER_LOCKS[interpreter_id]
    async with lock:
        worker = _get_worker(interpreter_id)
        result_dict = await asyncio.to_thread(
            worker.exec,
            payload.code,
            payload.tool_server_port,
            payload.tool_names,
            payload.session_id,
        )
        result: ExecResult = ExecResult.model_validate(result_dict)
        return result


@app.post("/cleanup")  # type: ignore
async def cleanup(payload: CleanupPayload) -> Dict[str, str]:
    interpreter_id = payload.interpreter_id
    worker = _get_worker(interpreter_id)
    worker.terminate()
    WORKERS.pop(interpreter_id, None)
    return {"status": "ok"}
