"""
# 配置文件，定义RAG系统中的一些参数
"""

RECALL_K = 10  # 检索时返回的文档数量
RERANK_TOP_K = 3  # 重新排序时返回的文档数量
BM25_WEIGHT = 0.5  # BM25检索结果的权重
VECTOR_WEIGHT = 0.5  # 向量检索结果的权重
COMPRESS_CONTEXT = (
    False  # 是否对检索到的文档进行压缩，提取与用户问题最相关的内容，减少上下文长度
)
