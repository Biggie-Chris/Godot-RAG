import os
import re
import json
import concurrent.futures
from typing import Dict, List, Tuple

from tqdm import tqdm
import tiktoken
import chromadb
from pathlib import Path


from Embeddings import get_embeddings

# 1.配置路径
DOC_DIR = "doc"
DOC_SOURCE_DIR = os.path.join(DOC_DIR, "_sources")
SEARCH_INDEX_JS = os.path.join(DOC_DIR, "searchindex.js")

DATA_DIR = "data"
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
META_DIR = os.path.join(DATA_DIR, "meta")
VECTORDB_DIR = os.path.join(DATA_DIR, "VectorDB")

CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")
SEARCHINDEX_JSON = os.path.join(META_DIR, "searchindex.json")

# 2.配置分词器
enc = tiktoken.get_encoding("cl100k_base")

def ensure_dirs() -> None:
    """创建所需目录"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    print("✅ 所有必要的目录已创建")
    

# 3.解析searchindex.js
def parse_searchindex() -> Dict:
    """
    将 searchindex.js 解析为 JSON 并保存。
    返回解析后的 dict
    """
    text = Path(SEARCH_INDEX_JS).read_text(encoding="utf-8")

    start = text.find("Search.setIndex(")
    if start == -1:
        raise RuntimeError("在 searchindex.js 中未找到 Search.setIndex(...) 结构")

    start = text.find("{", start)
    if start == -1:
        raise RuntimeError("Search.setIndex 后找不到 JSON 起始 {")

    # 匹配 JSON 花括号
    brace_count = 0
    end = start

    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    json_text = text[start:end + 1]

    # 解析 JSON
    data = json.loads(json_text)

    Path(SEARCHINDEX_JSON).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ 已写入 {SEARCHINDEX_JSON}")

    return data

# 4.建立 id 和 文档名的映射文件
def build_source_meta(searchindex: Dict) -> Dict[str, Dict]:
    """
    基于 searchindex.json，构建 doc_id → 源文件路径 的映射。
    Sphinx 的 searchindex.js 里 docnames 的顺序与文档 ID 一一对应。
    """
    """
    根据 searchindex.json 生成：
    {
        "0": { "name": "404", "source": "doc/_sources/404.txt" },
        ...
    }
    """

    print("📌 构建 source meta ...")

    src_map = {}

    # searchindex["filenames"] 存的就是相对路径，比如：
    # ["404", "about/complying_with_licenses", ...]
    filenames = searchindex.get("filenames", [])

    for idx, rel in enumerate(filenames):
        # 构造真实 txt 文件路径
        txt_path = Path(DOC_SOURCE_DIR) / (rel + ".txt")

        src_map[str(idx)] = {
            "name": rel,
            "source": str(txt_path),
        }

    # 保存 meta
    Path(os.path.join(META_DIR, "source_meta.json")).write_text(
        json.dumps(src_map, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ source_meta.json 已生成")

    return src_map
    
# 5. 遍历所有 txt 文件，并且生成 chunks
def iter_source_txt_files(root: str):
    """遍历 DOC_SOURCE_DIR 下的所有 .txt 文件"""
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".txt"):
                yield os.path.join(dirpath, fname)

def chunk_text(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 150,
) -> List[str]:
    
    tokens = enc.encode(text)
    chunks = []

    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + max_tokens, n)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        # 下一段起点 = 当前 start + (max_tokens - overlap_tokens)
        start += max_tokens - overlap_tokens

    return chunks

def generate_chunks(source_meta_map: Dict[str, Dict]) -> None:
    """从所有源 TXT 生成文本 chunk 并写入 chunks.jsonl"""
    
    """"
    每个 chunk 的格式: {"id": "0_3", "doc_id": "0", "text": "...内容...", "source": "doc/_sources/xxx.txt"}
    """
    fout = open(CHUNKS_FILE, "w", encoding="utf-8")
    print("开始生成 chunks ...")

    for doc_id, info in tqdm(source_meta_map.items()):
        # 使用 Path 构造真实文件路径，避免字符串错误
        src_file = Path(info["source"])  

        if not src_file.exists():
            print(f"文件不存在：{src_file}")
            continue

        text = src_file.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            record = {
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "text": chunk,
                "source": str(src_file),   # 保留原始字符串格式
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout.close()
    print(f"✅ 已生成 chunks 到 {CHUNKS_FILE}")
    
def build_chroma_from_chunks() -> None:
    """读取 chunks.jsonl，构建 ChromaDB 向量数据库（批量调用硅基流动 embedding）"""

    print("📌 开始构建 ChromaDB ...")

    client = chromadb.PersistentClient(path=VECTORDB_DIR)

    collection = client.get_or_create_collection(
        name="godot_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # 先把所有行读进内存，方便批量处理
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    print(f"需要生成向量的 chunks 数量：{total}")

    batch_size = 32  

    from math import ceil
    num_batches = ceil(total / batch_size)

    for bi in tqdm(range(num_batches), desc="Embedding (batch)"):
        start = bi * batch_size
        end = min(start + batch_size, total)
        batch_lines = lines[start:end]

        ids = []
        texts = []
        metas = []

        for line in batch_lines:
            item = json.loads(line)
            ids.append(item["id"])
            texts.append(item["text"])
            metas.append({"source": item["source"], "doc_id": item["doc_id"]})

        # 调用硅基流动批量 embedding
        try:
            vectors = get_embeddings(texts)
        except Exception as e:
            print(f"❌ 批量 embedding 出错，第 {bi} 批，跳过。错误：{e}")
            continue

        # 写入 ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metas,
        )

    print("ChromaDB 构建完成")


def main():
    ensure_dirs()
    
    # searchindex =  parse_searchindex()
    # meata_map = build_source_meta(searchindex)
    # generate_chunks(meata_map)
    build_chroma_from_chunks()
    
if __name__ == "__main__":
    main()