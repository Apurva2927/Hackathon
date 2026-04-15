import json
import logging
import os
import time
import uuid
from fastapi import FastAPI, Depends, HTTPException, Request, Query, Header
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from database import engine, get_db, Base
from models import Lead
from llm import score_lead_with_llm

# Create tables on startup
Base.metadata.create_all(bind=engine)


def _ensure_schema_compatibility() -> None:
    inspector = inspect(engine)
    if "leads" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("leads")}
    ddl = []
    if "llm_model" not in existing_columns:
        ddl.append("ALTER TABLE leads ADD COLUMN llm_model VARCHAR(120)")
    if "llm_confidence" not in existing_columns:
        ddl.append("ALTER TABLE leads ADD COLUMN llm_confidence DOUBLE PRECISION")
    if "llm_token_usage" not in existing_columns:
        ddl.append("ALTER TABLE leads ADD COLUMN llm_token_usage INTEGER")
    if "llm_cached" not in existing_columns:
        ddl.append("ALTER TABLE leads ADD COLUMN llm_cached BOOLEAN NOT NULL DEFAULT false")

    if not ddl:
        return

    with engine.begin() as connection:
        for stmt in ddl:
            connection.execute(text(stmt))


_ensure_schema_compatibility()

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead_scorer.api")


def _log(level: int, event: str, **kwargs) -> None:
    payload = {"event": event, **kwargs}
    logger.log(level, json.dumps(payload, default=str))


class LeadCreate(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    company: str = Field(min_length=2, max_length=255)
    description: str = Field(min_length=5, max_length=4000)


class LeadResponse(BaseModel):
    lead_id: int
    name: str
    company: str
    description: str
    score: int | None
    score_reason: str | None
    llm_model: str | None
    llm_confidence: float | None
    llm_token_usage: int | None
    llm_cached: bool
    created_at: str | None


class PaginatedLeadsResponse(BaseModel):
    items: list[LeadResponse]
    total: int
    limit: int
    offset: int


RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "60")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
_rate_limit_store: dict[str, list[float]] = {}


def _serialize_lead(lead: Lead) -> LeadResponse:
    return LeadResponse(
        lead_id=lead.id,
        name=lead.name,
        company=lead.company,
        description=lead.description,
        score=lead.score,
        score_reason=lead.score_reason,
        llm_model=lead.llm_model,
        llm_confidence=lead.llm_confidence,
        llm_token_usage=lead.llm_token_usage,
        llm_cached=bool(lead.llm_cached),
        created_at=lead.created_at.isoformat() if lead.created_at else None,
    )


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected_api_key = os.getenv("APP_API_KEY", "").strip()
    if not expected_api_key:
        return
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

app = FastAPI(
    title="Lead Scorer API",
    description="AI-powered lead scoring agent - Bug Slayers (Apurva Gunjal, Kunal Singh)",
    version="2.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    recent = _rate_limit_store.get(client_ip, [])
    recent = [ts for ts in recent if now - ts <= RATE_LIMIT_WINDOW_SECONDS]
    if len(recent) >= RATE_LIMIT_REQUESTS:
        _log(logging.WARNING, "rate_limit", request_id=request_id, client_ip=client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "request_id": request_id},
        )

    recent.append(now)
    _rate_limit_store[client_ip] = recent

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        _log(logging.ERROR, "middleware_exception", request_id=request_id)
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "n/a")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "request_id": request_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "n/a")
    _log(logging.ERROR, "unhandled_exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "request_id": request_id},
    )


@app.post("/leads", response_model=LeadResponse)
async def score_lead(
    payload: LeadCreate,
    request: Request,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Accept a lead, store it, score it via LLM, and return the result."""
    request_id = getattr(request.state, "request_id", "n/a")

    # 1. Create lead record
    lead = Lead(name=payload.name, company=payload.company, description=payload.description)
    db.add(lead)
    db.flush()  # get the ID before committing

    # 2. Call LLM for scoring
    result = score_lead_with_llm(
        name=payload.name,
        company=payload.company,
        description=payload.description,
        request_id=request_id,
    )
    lead.score = result["score"]
    lead.score_reason = result["reason"]
    lead.llm_model = result.get("model")
    lead.llm_confidence = result.get("confidence")
    lead.llm_token_usage = result.get("token_usage")
    lead.llm_cached = bool(result.get("cached", False))

    # 3. Commit to DB
    db.commit()
    db.refresh(lead)
    _log(logging.INFO, "lead_scored", request_id=request_id, lead_id=lead.id, score=lead.score)
    return _serialize_lead(lead)


@app.get("/leads", response_model=PaginatedLeadsResponse)
async def get_leads(
    request: Request,
    _: None = Depends(require_api_key),
    min_score: int | None = Query(default=None, ge=1, le=10),
    max_score: int | None = Query(default=None, ge=1, le=10),
    company: str | None = Query(default=None, min_length=1, max_length=255),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Return paginated leads sorted by score (highest first)."""
    query = db.query(Lead)
    if min_score is not None:
        query = query.filter(Lead.score >= min_score)
    if max_score is not None:
        query = query.filter(Lead.score <= max_score)
    if company:
        query = query.filter(Lead.company.ilike(f"%{company}%"))

    total = query.count()
    leads = query.order_by(Lead.score.desc(), Lead.created_at.desc()).offset(offset).limit(limit).all()

    request_id = getattr(request.state, "request_id", "n/a")
    _log(logging.INFO, "leads_listed", request_id=request_id, total=total, limit=limit, offset=offset)
    return PaginatedLeadsResponse(
        items=[_serialize_lead(lead) for lead in leads],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database not ready: {exc}") from exc

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.lower() in {"your_groq_api_key_here", "changeme", "replace_me"}:
        raise HTTPException(status_code=503, detail="LLM key is not configured")

    return {"status": "ready"}
