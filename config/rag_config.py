"""
# 配置文件，定义RAG系统中的一些参数
"""

CHUNK_SIZE = 500  # 文本切分的块大小，单位为字符
CHUNK_OVERLAP = 50  # 文本切分时块之间的重叠部分大小，单位为字符，确保上下文连续性

RECALL_K = 10  # 检索时返回的文档数量
RERANK_TOP_K = 3  # 重新排序时返回的文档数量

BM25_WEIGHT = 0.5  # BM25检索结果的权重
VECTOR_WEIGHT = 0.5  # 向量检索结果的权重

COMPRESS_CONTEXT = (
    False  # 是否对检索到的文档进行压缩，提取与用户问题最相关的内容，减少上下文长度
)
REWRITE_QUERY = None  # 问题重写规则 None表示不重写，"rule"表示使用规则进行重写，"llm"表示使用LLM进行重写

DOCS_DIR = "./docs" # 存放文档的目录，RAG系统将从这个目录中加载文档进行处理
CHROMA_DIR = "./chroma_db" # ChromaDB的存储目录