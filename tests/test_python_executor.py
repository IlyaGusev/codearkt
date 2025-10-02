import pytest

from academia_mcp.tools import show_image

from codearkt.python_executor import PythonExecutor, ExecResult

from tests.conftest import MCPServerTest

SNIPPET_1 = """
answer = "Answer 1"
print(answer, end="")
"""

SNIPPET_2 = """
print("Variable still here:", answer, end="")
"""

SNIPPET_3 = """
import json
answer1 = arxiv_download(paper_id="2506.15003")["title"]
print("Answer 1:", answer1, end="")
"""

SNIPPET_4 = """
import json
doc = arxiv_download(paper_id="2506.15003")
answer = document_qa(question="What is the capital of France?", document=json.dumps(doc))
print(answer, end="")
"""

SNIPPET_5 = """
answer = arxiv_download(paper_id="2506.15003")
print(answer["abstract"], end="")
"""

EXPR_SNIPPET_1 = """
a = 1
a
"""

EXPR_SNIPPET_2 = """
arxiv_download(paper_id="2506.15003")
"""

BAD_SNIPPET_1 = """from dfdaf import dfadfa

print("Hello!")
"""

EXCEPT_SNIPPET_1 = """
1 / 0
"""

PIP_INSTALL_SNIPPET = """
import subprocess
proc = subprocess.run(['pip', 'install', 'transformers'], capture_output=True)
print(proc.stdout)
print(proc.stderr)
"""

ASSIGN_SNIPPET_1 = """
a = 1
"""


class TestPythonExecutor:
    async def test_python_executor_basic(self) -> None:
        python_executor = PythonExecutor()
        result1 = await python_executor.ainvoke(SNIPPET_1)
        result2 = await python_executor.ainvoke(SNIPPET_2)
        assert result1.stdout == "Answer 1"
        assert result2.stdout == "Variable still here: Answer 1"

    async def test_python_executor_pip_install(self) -> None:
        python_executor = PythonExecutor()
        result = await python_executor.ainvoke(PIP_INSTALL_SNIPPET)
        assert "ERROR" in result.stdout, result.stdout

    async def test_python_executor_result_expr(self) -> None:
        python_executor = PythonExecutor()
        result = await python_executor.ainvoke("r = 4\nr")
        assert result.result == 4

    async def test_python_executor_result_dict(self) -> None:
        python_executor = PythonExecutor()
        result = await python_executor.ainvoke("r = {'a': 4}\nr")
        assert result.result == {"a": 4}

    async def test_python_executor_mcp_call_no_tools(self, mcp_server_test: MCPServerTest) -> None:
        _ = mcp_server_test
        python_executor = PythonExecutor(
            tools_server_host=mcp_server_test.host,
            tools_server_port=mcp_server_test.port,
        )
        result = await python_executor.ainvoke(SNIPPET_3)
        assert (
            result.stdout
            != "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"
        )
        assert result.error
        assert "Error" in result.error

    async def test_python_executor_mcp_call_tool(
        self,
        mcp_server_test: MCPServerTest,
    ) -> None:
        _ = mcp_server_test
        executor = PythonExecutor(
            tool_names=["arxiv_download"],
            tools_server_host=mcp_server_test.host,
            tools_server_port=mcp_server_test.port,
        )
        result = await executor.ainvoke(SNIPPET_3)
        assert (
            result.stdout
            == "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"
        ), str(result)

    async def test_python_executor_non_existing_tool(self) -> None:
        executor = PythonExecutor(tool_names=["arxiv_download"])
        with pytest.raises(ValueError):
            await executor.ainvoke(SNIPPET_3)

    async def test_python_executor_mcp_document_qa(
        self,
        mcp_server_test: MCPServerTest,
    ) -> None:
        _ = mcp_server_test
        executor = PythonExecutor(
            tool_names=["arxiv_download", "document_qa"],
            tools_server_host=mcp_server_test.host,
            tools_server_port=mcp_server_test.port,
        )
        result = await executor.ainvoke(SNIPPET_4)
        assert result.stdout and "Error" not in result.stdout, str(result)

    async def test_python_executor_bad_snippet(self) -> None:
        executor = PythonExecutor()
        result = await executor.ainvoke(BAD_SNIPPET_1)
        assert result.error
        assert "ModuleNotFoundError" in result.error
        assert "line 1" in result.error

    async def test_python_executor_expressions(
        self,
        mcp_server_test: MCPServerTest,
    ) -> None:
        _ = mcp_server_test
        executor = PythonExecutor(
            tool_names=["arxiv_download"],
            tools_server_host=mcp_server_test.host,
            tools_server_port=mcp_server_test.port,
        )
        result = await executor.ainvoke(EXPR_SNIPPET_1)
        assert result.result == 1
        result = await executor.ainvoke(EXPR_SNIPPET_2)
        assert result.result is not None

    async def test_python_executor_exceptions(self) -> None:
        executor = PythonExecutor()
        result = await executor.ainvoke(EXCEPT_SNIPPET_1)
        assert result.error
        assert "ZeroDivisionError" in result.error
        assert "line 2" in result.error

    async def test_python_executor_assign_to_variable(self) -> None:
        executor = PythonExecutor()
        result = await executor.ainvoke(ASSIGN_SNIPPET_1)
        assert result.result == 1

    async def test_python_executor_mcp_call_structured_tool(
        self,
        mcp_server_test: MCPServerTest,
    ) -> None:
        _ = mcp_server_test
        executor = PythonExecutor(
            tool_names=["arxiv_download"],
            tools_server_host=mcp_server_test.host,
            tools_server_port=mcp_server_test.port,
        )
        result = await executor.ainvoke(SNIPPET_5)
        assert result.stdout.startswith("We present evidence")


def test_execresult_to_message_with_text_result() -> None:
    r = ExecResult(stdout="hello", error=None, result={"a": 1})
    msg = r.to_message()
    assert msg.role == "user"
    assert isinstance(msg.content, list)
    text_parts = [c for c in msg.content if c.get("type") == "text"]
    assert text_parts and "Output:" in text_parts[0]["text"]


def test_execresult_to_message_with_image(test_image_url: str) -> None:
    r = ExecResult(stdout="", error=None, result=show_image(test_image_url))
    msg = r.to_message()
    assert isinstance(msg.content, list)
    assert any(c.get("type") == "image_url" for c in msg.content)
