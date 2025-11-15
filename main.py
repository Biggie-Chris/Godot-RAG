import json
import chromadb
from LLM import OpenAIChat
from Embeddings import get_embeddings

VECTORDB_DIR = "data/VectorDB"


# 1 连接 Chroma
def get_chroma_collection():
    client = chromadb.PersistentClient(path=VECTORDB_DIR)
    return client.get_or_create_collection(
        name="godot_docs",
        metadata={"hnsw:space": "cosine"}
    )


# 2 RAG 检索
def retrieve_docs(query: str, top_k: int = 5):
    collection = get_chroma_collection()
    emb = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    return results


# 3 构建 context（把 chunk 拼起来）
def build_context(results):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context_parts = []
    for text, meta in zip(docs, metas):
        context_parts.append(f"[doc_id={meta.get('doc_id')}] 来自文件：{meta.get('source')}\n{text}")

    return "\n\n".join(context_parts)


# 4 RAG 主逻辑（返回答案 + 引用来源）
def answer_with_rag(query: str, top_k: int = 5):
    results = retrieve_docs(query, top_k)
    context = build_context(results)

    llm = OpenAIChat()
    history = []

    final_answer = llm.chat(
        prompt=query,
        history=history,
        content=context
    )

    # 解析引用
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    citations = []
    for meta, dist in zip(metas, dists):
        citations.append({
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "score": round(1 - float(dist), 4)  # 相似度
        })

    return final_answer, citations


# 5 命令行 REPL
def repl():
    print("🎮 Godot RAG 系统启动！输入你的问题，输入 exit 退出。")

    while True:
        q = input("\n❓ 你的问题： ").strip()
        if q.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break

        print("🔍 正在检索 + 生成回答 ...")

        try:
            answer, refs = answer_with_rag(q)

            print("\n💬 回答：\n")
            print(answer)

            print("\n📚 引用来源：")
            for r in refs:
                print(f"- doc_id: {r['doc_id']} | 相似度: {r['score']}")
                print(f"  ↳ {r['source']}")

            print("\n-------------------------------------------")

        except Exception as e:
            print(f"❌ 出错：{e}")


if __name__ == "__main__":
    repl()
