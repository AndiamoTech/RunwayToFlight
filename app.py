# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RunwayToLiftoff import compute, build_outputs, coerce_date

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
    accent_colors: str = "#12c04c"

app = FastAPI(title="RunwayToFlight API", version="0.1.0")

@app.get("/")
def home():
    return {"message": "RunwayToFlight API is live 🚀 — POST /runway with JSON. See /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/runway")
def runway(p: Payload):
    try:
        data = p.model_dump()
        data["formation_date"] = coerce_date(data["formation_date"])
        calc = compute(data)
        prompt, summary = build_outputs(data, calc)
        return {"prompt": prompt, "summary": summary, "calc": calc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))