from typing import Any, Callable, Dict, Optional
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.client.transports import ClientTransport, SSETransport, StreamableHttpTransport
from fastmcp.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
    infer_transport_type_from_url,
)
from codearkt.settings import settings


def get_mcp_app(
    mcp_config: Optional[Dict[str, Any]],
    additional_tools: Optional[Dict[str, Callable[..., Any]]] = None,
    add_prefixes: bool = True,
) -> FastAPI:
    mcp: FastMCP[Any] = FastMCP(name="Codearkt MCP Proxy")
    if mcp_config:
        cfg = MCPConfig.from_dict(mcp_config)
        server_count = len(cfg.mcpServers)

        for name, server in cfg.mcpServers.items():
            transport: Optional[ClientTransport] = None
            if isinstance(server, RemoteMCPServer):
                transport_type = server.transport or infer_transport_type_from_url(server.url)
                if transport_type == "sse":
                    transport = SSETransport(
                        server.url,
                        headers=server.headers,
                        auth=server.auth,
                        sse_read_timeout=settings.PROXY_SSE_READ_TIMEOUT,
                    )
                else:
                    transport = StreamableHttpTransport(
                        server.url,
                        headers=server.headers,
                        auth=server.auth,
                        sse_read_timeout=settings.PROXY_SSE_READ_TIMEOUT,
                    )
            elif isinstance(server, StdioMCPServer):
                transport = server.to_transport()

            assert transport is not None, "Transport is required for the MCP server in the config"
            sub_proxy = FastMCP.as_proxy(backend=transport)
            prefix: Optional[str] = None if server_count == 1 else name
            if not add_prefixes:
                prefix = None
            mcp.mount(prefix=prefix, server=sub_proxy)

    if additional_tools:
        for name, tool in additional_tools.items():
            mcp.tool(tool, name=name)

    return mcp.http_app()
