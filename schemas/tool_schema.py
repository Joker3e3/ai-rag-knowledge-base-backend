from pydantic import BaseModel


class ToolQuery(BaseModel):
    user_id: str
    candidate_id: str
    resume_id: str
    question: str