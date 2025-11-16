import asyncio
from fastmcp import FastMCP
from typing import Any, Dict

from main import answer_with_rag   

# 创建 MCP 服务
server = FastMCP("rag-mcp-server")


@server.tool(
    name="rag.query",
    description=(
        "从本地 Godot 文档向量数据库检索信息并回答用户的 Godot 相关问题。"
        "包括节点(Node)、脚本(GDScript)、场景(Scene)、API、信号(Signal)、物理(Physics)等内容。"
    )
)
async def rag_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    answer, citations = answer_with_rag(query, top_k)
    return {
        "answer": answer,
        "citations": citations
    }


async def main():
    await server.run_stdio_async()  


if __name__ == "__main__":
    asyncio.run(main())
