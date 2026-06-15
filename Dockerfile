FROM python:3.11-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-prod.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]