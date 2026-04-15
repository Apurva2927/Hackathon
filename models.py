from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, func
from database import Base


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    score = Column(Integer, nullable=True)
    score_reason = Column(Text, nullable=True)
    llm_model = Column(String(120), nullable=True)
    llm_confidence = Column(Float, nullable=True)
    llm_token_usage = Column(Integer, nullable=True)
    llm_cached = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
