# app.py — RunwayToFlight API v3.5
import os
from typing import List, Optional
from datetime import datetime, date

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from runwaytoflight import compute, build_prompt, build_summary, coerce_date


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
    loan_apr_pct: float = 3.0
    loan_term_years: float = 3.0
    accent_colors: str = "#12c04c"
    start_date: Optional[str] = None  # Optional base date for simulation


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
    Enforce x-api-key when RUNWAY_API_KEY is set.
    If no key(s) configured, endpoint remains open.
    """
    allowed = _load_api_keys()
    if not allowed:
        return

    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


# -----------------------------
# Helpers
# -----------------------------
def _parse_base_date(start: Optional[str]) -> Optional[date]:
    """
    Map optional start_date string → date, using the same coerce_date
    behavior as the CLI. If None or invalid, return None and let
    runwaytoflight fall back internally.
    """
    if not start:
        return None
    try:
        norm = coerce_date(start)           # normalize YYYY-MM → YYYY-MM-01
        dt = datetime.strptime(norm, "%Y-%m-%d")
        return dt.date()
    except Exception:
        return None


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="RunwayToFlight API", version="3.4.0")


@app.get("/")
def home():
    return {
        "message": "RunwayToFlight API is live 🚀 — POST /runway with JSON. See /docs for Swagger UI.",
        "auth": "Send x-api-key header if RUNWAY_API_KEY is configured.",
        "version": "3.4.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runway", dependencies=[Depends(require_api_key)])
def runway(p: Payload):
    """
    Compute runway, breakeven, and funding gap metrics.
    Returns: prompt, summary, and raw calculation dictionary.
    """
    try:
        data = p.model_dump()
        # Normalize formation date
        data["formation_date"] = coerce_date(data["formation_date"])
        # Convert optional start_date → base_date for the simulator
        base_date = _parse_base_date(data.pop("start_date", None))

        calc = compute(data, base_date=base_date)
        prompt = build_prompt(data, calc)
        summary = build_summary(calc)
        return {"prompt": prompt, "summary": summary, "calc": calc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
