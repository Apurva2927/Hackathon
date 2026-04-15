import json
import logging
import os
import time
import uuid
from io import BytesIO, StringIO
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request, Query, Header, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from database import engine, get_db, Base
from models import Lead, Followup
from llm import score_lead_with_llm, generate_followup_email_with_llm

# Create tables on startup
Base.metadata.create_all(bind=engine)


def _ensure_schema_compatibility() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    ddl = []

    if "leads" in table_names:
        existing_columns = {col["name"] for col in inspector.get_columns("leads")}
        if "llm_model" not in existing_columns:
            ddl.append("ALTER TABLE leads ADD COLUMN llm_model VARCHAR(120)")
        if "llm_confidence" not in existing_columns:
            ddl.append("ALTER TABLE leads ADD COLUMN llm_confidence DOUBLE PRECISION")
        if "llm_token_usage" not in existing_columns:
            ddl.append("ALTER TABLE leads ADD COLUMN llm_token_usage INTEGER")
        if "llm_cached" not in existing_columns:
            ddl.append("ALTER TABLE leads ADD COLUMN llm_cached BOOLEAN NOT NULL DEFAULT false")

    if "followups" in table_names:
        existing_columns = {col["name"] for col in inspector.get_columns("followups")}
        if "lead_id" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN lead_id INTEGER")
        if "lead_name" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN lead_name VARCHAR(255)")
        if "company" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN company VARCHAR(255)")
        if "description" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN description TEXT")
        if "days_since_last_interaction" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN days_since_last_interaction INTEGER")
        if "email_text" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN email_text TEXT")
        if "llm_model" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN llm_model VARCHAR(120)")
        if "llm_token_usage" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN llm_token_usage INTEGER")
        if "llm_cached" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN llm_cached BOOLEAN NOT NULL DEFAULT false")
        if "created_at" not in existing_columns:
            ddl.append("ALTER TABLE followups ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT now()")

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


class BulkUploadError(BaseModel):
    row: int
    error: str


class BulkLeadUploadResponse(BaseModel):
    total_rows: int
    created_count: int
    failed_count: int
    leads: list[LeadResponse]
    errors: list[BulkUploadError]


class FollowupGenerateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    company: str = Field(min_length=2, max_length=255)
    description: str = Field(min_length=5, max_length=4000)
    days_since_last_interaction: int = Field(ge=0, le=3650)
    lead_id: int | None = Field(default=None, ge=1)


class FollowupResponse(BaseModel):
    followup_id: int
    lead_id: int | None
    lead_name: str
    company: str
    days_since_last_interaction: int
    email_text: str
    llm_model: str | None
    llm_token_usage: int | None
    llm_cached: bool
    created_at: str | None


RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "60")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
MAX_BULK_ROWS = max(1, int(os.getenv("MAX_BULK_UPLOAD_ROWS", "200")))
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


def _serialize_followup(followup: Followup) -> FollowupResponse:
    return FollowupResponse(
        followup_id=followup.id,
        lead_id=followup.lead_id,
        lead_name=followup.lead_name,
        company=followup.company,
        days_since_last_interaction=followup.days_since_last_interaction,
        email_text=followup.email_text,
        llm_model=followup.llm_model,
        llm_token_usage=followup.llm_token_usage,
        llm_cached=bool(followup.llm_cached),
        created_at=followup.created_at.isoformat() if followup.created_at else None,
    )


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected_api_key = os.getenv("APP_API_KEY", "").strip()
    if not expected_api_key:
        return
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _parse_lead_dataframe(file_name: str, content: bytes) -> pd.DataFrame:
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    lower_name = (file_name or "").lower()
    if lower_name.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(content), engine="openpyxl")
    elif lower_name.endswith(".csv"):
        decoded = content.decode("utf-8-sig")
        df = pd.read_csv(StringIO(decoded))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .xlsx or .csv")

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows found in uploaded file")

    df.columns = [str(col).strip().lower() for col in df.columns]
    required = ["name", "company", "description"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    return df[required]

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


@app.post("/leads/bulk-upload", response_model=BulkLeadUploadResponse)
async def bulk_upload_leads(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Upload multiple leads from Excel/CSV and score each row."""
    request_id = getattr(request.state, "request_id", "n/a")
    content = await file.read()
    df = _parse_lead_dataframe(file.filename or "", content)

    if len(df) > MAX_BULK_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows ({len(df)}). Maximum allowed is {MAX_BULK_ROWS}.",
        )

    leads: list[LeadResponse] = []
    errors: list[BulkUploadError] = []

    for idx, row in df.iterrows():
        row_number = int(idx) + 2
        raw_name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
        raw_company = "" if pd.isna(row["company"]) else str(row["company"]).strip()
        raw_description = "" if pd.isna(row["description"]) else str(row["description"]).strip()

        try:
            payload = LeadCreate(name=raw_name, company=raw_company, description=raw_description)
            result = score_lead_with_llm(
                name=payload.name,
                company=payload.company,
                description=payload.description,
                request_id=request_id,
            )

            lead = Lead(name=payload.name, company=payload.company, description=payload.description)
            lead.score = result["score"]
            lead.score_reason = result["reason"]
            lead.llm_model = result.get("model")
            lead.llm_confidence = result.get("confidence")
            lead.llm_token_usage = result.get("token_usage")
            lead.llm_cached = bool(result.get("cached", False))

            db.add(lead)
            db.commit()
            db.refresh(lead)
            leads.append(_serialize_lead(lead))
        except Exception as exc:
            db.rollback()
            errors.append(BulkUploadError(row=row_number, error=str(exc)))

    _log(
        logging.INFO,
        "bulk_upload_completed",
        request_id=request_id,
        total_rows=len(df),
        created_count=len(leads),
        failed_count=len(errors),
    )
    return BulkLeadUploadResponse(
        total_rows=len(df),
        created_count=len(leads),
        failed_count=len(errors),
        leads=leads,
        errors=errors,
    )


@app.post("/followups/generate", response_model=FollowupResponse)
async def generate_followup(
    payload: FollowupGenerateRequest,
    request: Request,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Generate a personalized follow-up email and persist it."""
    request_id = getattr(request.state, "request_id", "n/a")

    result = generate_followup_email_with_llm(
        name=payload.name,
        company=payload.company,
        description=payload.description,
        days_since_last_interaction=payload.days_since_last_interaction,
        request_id=request_id,
    )

    followup = Followup(
        prospect=payload.name,
        last_interaction=f"{payload.days_since_last_interaction} days ago",
        days_since=payload.days_since_last_interaction,
        email=result["email_text"],
        lead_id=payload.lead_id,
        lead_name=payload.name,
        company=payload.company,
        description=payload.description,
        days_since_last_interaction=payload.days_since_last_interaction,
        email_text=result["email_text"],
        llm_model=result.get("model"),
        llm_token_usage=result.get("token_usage"),
        llm_cached=bool(result.get("cached", False)),
    )
    db.add(followup)
    db.commit()
    db.refresh(followup)

    _log(
        logging.INFO,
        "followup_generated",
        request_id=request_id,
        followup_id=followup.id,
        lead_id=followup.lead_id,
    )
    return _serialize_followup(followup)


@app.get("/followups", response_model=list[FollowupResponse])
async def list_followups(
    _: None = Depends(require_api_key),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    followups = (
        db.query(Followup)
        .filter(Followup.lead_name.isnot(None))
        .filter(Followup.company.isnot(None))
        .filter(Followup.days_since_last_interaction.isnot(None))
        .filter(Followup.email_text.isnot(None))
        .order_by(Followup.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_serialize_followup(item) for item in followups]


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
