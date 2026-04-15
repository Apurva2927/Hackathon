from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, func, ForeignKey
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


class Followup(Base):
    __tablename__ = "followups"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Legacy columns kept for backward compatibility with existing databases.
    prospect = Column(String(100), nullable=False)
    last_interaction = Column(Text, nullable=False)
    days_since = Column(Integer, nullable=False)
    email = Column(Text, nullable=False)

    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=True, index=True)
    lead_name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    days_since_last_interaction = Column(Integer, nullable=False)
    email_text = Column(Text, nullable=False)
    llm_model = Column(String(120), nullable=True)
    llm_token_usage = Column(Integer, nullable=True)
    llm_cached = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
