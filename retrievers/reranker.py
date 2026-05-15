
# import torch
# print("CUDA 可用性：", torch.cuda.is_available())      # ROCm 下也返回 True
# print("ROCm 版本：", torch.version.hip)              # 有值说明是 ROCm 版，None 说明是普通 CUDA 版

# rerank_model = CrossEncoder(
#     "BAAI/bge-reranker-base"
# )

# rerank 模型懒加载
rerank_model = None


def get_rerank_model():
    global rerank_model

    if rerank_model is None:
        print("正在加载 reranker 模型...")

        from FlagEmbedding import FlagReranker

        rerank_model = FlagReranker(
            "./models/bge-reranker-v2-m3",
            use_fp16=False,
            use_fast_tokenizer=True,
            device="cpu",
        )

        print("reranker 模型加载完成")

    return rerank_model


def rerank_documents(query, docs, top_k=3):

    if not docs:
        return []

    model = get_rerank_model()

    # [问题, 内容]
    pairs = [[query, doc.page_content] for doc in docs]

    scores = model.compute_score(pairs)
    try:
        scores = list(scores)
    except TypeError:
        scores = [scores]

    # [文档, 分数]
    scored_docs = list(zip(docs, scores))

    # 根据分数降序排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_k]
