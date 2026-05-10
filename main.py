from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 创建客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 调用模型
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "你是一个AI助手"
        },
        {
            "role": "user",
            "content": "请解释什么是RAG"
        }
    ]
)

# 输出结果
print(response.choices[0].message.content)