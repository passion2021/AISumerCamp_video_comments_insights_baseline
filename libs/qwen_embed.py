from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
from loguru import logger


class QwenEmbeddingTransformer:
    """本地加载 Qwen Embedding 模型测试使用"""

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path, device="cuda:0")

    def fit(self, X, y=None):
        # 预训练模型无需拟合数据
        return self

    def transform(self, X):
        if isinstance(X, str):
            texts = [X]
        else:
            texts = X
        return self.model.encode(texts)

if __name__ == '__main__':
    embedder = QwenEmbeddingTransformer(r"E:\ai-eb\models\Qwen3-Embedding-0.6B")
    # 没有 prompt_name（文档类型）
    doc_emb = embedder.transform(["Beijing is the capital of China."])
    # 输出 shape
    print("Doc embedding shape:", doc_emb.shape)
