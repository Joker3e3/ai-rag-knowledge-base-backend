from database.database import engine, Base

# 必须 import 才会注册到 Base.metadata
from database.models.candidate import Candidate
from database.models.resume import Resume


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ RAG database initialized successfully")


if __name__ == "__main__":
    init_db()