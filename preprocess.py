import os
import re
import json
from typing import Dict, List, Tuple

from tqdm import tqdm
import tiktoken
import chromadb
from pathlib import Path

from Embeddings import get_embedding 

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
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    print("✅所有必要的目录已创建")
    

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

# 4.
def build_source_meta(searchindex: Dict) -> Dict[str, Dict]:
    pass

# 5. 
def iter_source_txt_files(root: str):
    pass

def chunk_text(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 150,
) -> List[str]:
    pass

def generate_chunks(source_meta_map: Dict[str, Dict]) -> None:
    pass

def build_chroma_from_chunks() -> None:
    pass





def main():
    ensure_dirs()
    
    parse_searchindex()

if __name__ == "__main__":
    main()