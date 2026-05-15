import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from config.rag_config import CHROMA_DIR
from retrievers.bm25_store import rebuild_bm25_docs_from_chroma

load_dotenv()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

rebuild_bm25_docs_from_chroma(vectorstore)

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    temperature=0.3,
    streaming=True,
    timeout=60.0
)