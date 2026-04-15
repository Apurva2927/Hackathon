from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import engine, get_db, Base
from models import Lead
from llm import score_lead_with_llm
import os

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Lead Scorer API",
    description="AI-powered lead scoring agent – Bug Slayers (Apurva Gunjal, Kunal Singh)",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/leads")
async def score_lead(name: str, company: str, description: str, db: Session = Depends(get_db)):
    """Accept a lead, store it, score it via LLM, and return the result."""
    # 1. Create lead record
    lead = Lead(name=name, company=company, description=description)
    db.add(lead)
    db.flush()  # get the ID before committing

    # 2. Call LLM for scoring
    result = score_lead_with_llm(name=name, company=company, description=description)
    lead.score = result["score"]
    lead.score_reason = result["reason"]

    # 3. Commit to DB
    db.commit()
    db.refresh(lead)

    return {
        "lead_id": lead.id,
        "name": lead.name,
        "company": lead.company,
        "score": lead.score,
        "score_reason": lead.score_reason,
    }


@app.get("/leads")
async def get_leads(db: Session = Depends(get_db)):
    """Return all leads sorted by score (highest first)."""
    leads = db.query(Lead).order_by(Lead.score.desc()).all()
    return [
        {
            "id": l.id,
            "name": l.name,
            "company": l.company,
            "description": l.description,
            "score": l.score,
            "score_reason": l.score_reason,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in leads
    ]
