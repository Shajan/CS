from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("sdasan_server")


@mcp.tool()
async def special_greeting(name: str) -> str:
    """Provide a special greeting message

    Args:
        name: Name of the person who needs a special greeting
    """

    return f"Namaskaram {name}!\n"


@mcp.tool()
async def regular_greeting(name: str) -> str:
    """Provide regular greeting message

    Args:
        name: Name of the person who needs the greeting
    """

    return f"Hello {name}!\n"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
