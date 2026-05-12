from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI

import os

# 这里是加载环境变量，确保你已经在项目根目录下创建了一个 .env 文件，并在其中设置了 OPENAI_API_KEY
load_dotenv()

# 读取 PDF
loader = PyPDFLoader("./docs/test.pdf")

documents = loader.load()

# 进行文本切片

# 为什么必须切片？
# 因为：
# LLM：
# token 有限制
# embedding 太长效果差
# 企业 RAG 最重要问题之一：
# chunking strategy

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents)

# 创建 Embedding 模型
## 错误点：之前使用了 OpenAIEmbeddings，导致无法使用 Qwen 模型进行向量化
## 遇到问题，python版本太新，tiktok
# embeddings = OpenAIEmbeddings(
#     api_key=os.getenv("QWEN_API_KEY"),
#     base_url=os.getenv("QWEN_BASE_URL"),
#     model="text-embedding-v4"
# )
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 创建 语义搜索器
retriever = vectorstore.as_retriever()

# 创建 LLM 模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    # 控制随机性：0=最严谨，1=最有创意
    temperature=0.7,
    # 超时时间，避免卡死
    timeout=60.0
)

# 开始检索
query = "这个PDF主要讲了什么"

relevant_docs = retriever.invoke(query)

print("=== 检索结果 ===")

for doc in relevant_docs[:3]:
    print(doc.page_content)
    print("------")