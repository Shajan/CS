#pip install mcp

#!/usr/bin/env python3
import getpass
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math-and-username-tools")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def get_username() -> str:
    """Return the current OS username of the machine running this server."""
    return getpass.getuser()


if __name__ == "__main__":
    # Runs an MCP server over stdio (what most MCP clients expect)
    mcp.run()