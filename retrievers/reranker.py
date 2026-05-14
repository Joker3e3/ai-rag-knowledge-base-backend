from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker

# import torch
# print("CUDA 可用性：", torch.cuda.is_available())      # ROCm 下也返回 True
# print("ROCm 版本：", torch.version.hip)              # 有值说明是 ROCm 版，None 说明是普通 CUDA 版

# rerank_model = CrossEncoder(
#     "BAAI/bge-reranker-base"
# )
rerank_model = FlagReranker(
    "./models/bge-reranker-v2-m3",
    use_fp16=False,
    use_fast_tokenizer=True,
    device='cpu',
)

def rerank_documents(query, docs, top_k=3):

    if not docs:
        return []

    # [问题, 内容]
    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = rerank_model.compute_score(pairs)

    # [文档, 分数]
    scored_docs = list(zip(docs, scores))

    # 根据分数降序排序
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [
        doc for doc, score in scored_docs[:top_k]
    ]

    return scored_docs[:top_k]