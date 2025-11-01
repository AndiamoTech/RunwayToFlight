# app.py — RunwayToFlight API v3.3
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# Updated import: matches new filename and functions
from runwaytoflight import compute, build_outputs, coerce_date


# -----------------------------
# Models
# -----------------------------
class Payload(BaseModel):
    company_name: str
    formation_date: str           # "YYYY-MM" or "YYYY-MM-DD"
    mrr: float
    last_mrr: float
    mrc: float
    growth_pct: float
    cost_growth_pct: float
    bootstrap_cash: float
    external_equity_cash: float
    grant_cash: float
    loan_cash: float
    seed_mrr: float = 0.0
    accent_colors: str = "#12c04c"


# -----------------------------
# Auth helper
# -----------------------------
def _load_api_keys() -> List[str]:
    """
    Read RUNWAY_API_KEY from environment.
    Accepts a single key or comma-separated list.
    Whitespace around commas is ignored.
    """
    raw = os.getenv("RUNWAY_API_KEY", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Dependency that enforces x-api-key when RUNWAY_API_KEY is set.
    If no key(s) configured, endpoint remains open (useful during initial setup).
    """
    allowed = _load_api_keys()
    if not allowed:
        return  # No keys configured → allow (can flip this to block if preferred)

    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="RunwayToFlight API", version="3.3.0")


@app.get("/")
def home():
    return {
        "message": "RunwayToFlight API is live 🚀 — POST /runway with JSON. See /docs for Swagger UI.",
        "auth": "Send x-api-key header if RUNWAY_API_KEY is configured.",
        "version": "3.3.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runway", dependencies=[Depends(require_api_key)])
def runway(p: Payload):
    """
    Compute runway, breakeven, and funding gap metrics.
    Returns the same structure as CLI: prompt, summary, and raw calculation dictionary.
    """
    try:
        data = p.model_dump()
        data["formation_date"] = coerce_date(data["formation_date"])
        calc = compute(data)
        prompt, summary = build_outputs(data, calc)
        return {"prompt": prompt, "summary": summary, "calc": calc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))