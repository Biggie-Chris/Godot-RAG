import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path (str): 模型或数据的路径
            is_api (bool): 是否使用API方式。True表示使用在线API服务，False表示使用本地模型
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        获取文本的嵌入向量表示
        Args:
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError
    
    # @classmethod
    # def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
    #     """
    #     计算两个向量之间的余弦相似度
    #     Args:
    #         vector1 (List[float]): 第一个向量
    #         vector2 (List[float]): 第二个向量
    #     Returns:
    #         float: 两个向量的余弦相似度，范围在[-1,1]之间
    #     """
    #     # 将输入列表转换为numpy数组，并指定数据类型为float32
    #     v1 = np.array(vector1, dtype=np.float32)
    #     v2 = np.array(vector2, dtype=np.float32)

    #     # 检查向量中是否包含无穷大或NaN值
    #     if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
    #         return 0.0

    #     # 计算向量的点积
    #     dot_product = np.dot(v1, v2)
    #     # 计算向量的范数（长度）
    #     norm_v1 = np.linalg.norm(v1)
    #     norm_v2 = np.linalg.norm(v2)
        
    #     # 计算分母（两个向量范数的乘积）
    #     magnitude = norm_v1 * norm_v2
    #     # 处理分母为0的特殊情况
    #     if magnitude == 0:
    #         return 0.0
            
    #     # 返回余弦相似度
    #     return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embeddings(self, texts, model="Qwen/Qwen3-Embedding-4B"):
        if not self.is_api:
            raise NotImplementedError

        cleaned = [t.replace("\n", " ") for t in texts]
        resp = self.client.embeddings.create(input=cleaned, model=model)

        vectors = []
        for d in resp.data:
            emb = d.embedding

            # flatten 任意嵌套的 [[[vec]]] → [vec]
            while isinstance(emb, list) and len(emb) == 1:
                emb = emb[0]

            vectors.append(emb)

        return vectors


def get_embeddings(texts: List[str], model: str = "Qwen/Qwen3-Embedding-4B") -> List[List[float]]:
    """
    批量 embedding 的便捷函数
    """
    embedder = OpenAIEmbedding(is_api=True)
    return embedder.get_embeddings(texts, model)