from langchain.text_splitter import RecursiveCharacterTextSplitter

# 优先按段落、换行和中文标点切分，尽量保留简历条目的完整语义。
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；"], chunk_size=500, chunk_overlap=50
)


def split_chunks(documents):
    """把按 section 拆好的文档继续切成适合 embedding 的小块。"""
    return text_splitter.split_documents(documents)
