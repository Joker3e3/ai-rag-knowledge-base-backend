from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.chat_routes import router as chat_router
import uvicorn

from routes.document_routes import router as document_router
from routes.tool_routes import router as tool_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(document_router)
app.include_router(tool_router)

print("main.py 已加载")
# 启动 FastAPI 应用
if __name__ == "__main__":

    print("准备启动 uvicorn")
    import uvicorn

    # uvicorn.run 启动开发服务器
    # 参数：
    # app → 当前文件中的 FastAPI app 实例
    # host="0.0.0.0" → 允许外部访问
    # port=8000 → 监听端口
    # reload=False → 文件修改时不自动重启（生产模式用）
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
