"""Optional local MCP v1 example. Principal is fixed by trusted launcher configuration."""
import os
from mcp.server.fastmcp import FastMCP
from harbor_tools import dispatch
mcp = FastMCP("Harbor Metals")
principal = os.environ.get("HARBOR_DEMO_PRINCIPAL", "research")
@mcp.tool()
def search_documents(query: str) -> list:
    """Find Harbor Metals source passages."""
    return dispatch(principal,"search_documents",{"query":query})
@mcp.tool()
def calculate_leverage(debt: float, eligible_cash: float, ebitda: float) -> dict:
    """Compute leverage from eligible financial inputs."""
    return dispatch(principal,"calculate_leverage",locals())
@mcp.tool()
def release_brief(brief: str) -> dict:
    """Simulate releasing a brief; requires reviewer permission."""
    return dispatch(principal,"release_brief",{"brief":brief})
if __name__ == "__main__":
    mcp.run(transport="stdio")
