# 🛫 RunwayToFlight 

# Founder Runway & Liftoff Planner



**RunwayToFlight** is a lightweight Python tool and FastAPI microservice that helps founders model their startup’s runway, breakeven, and funding needs. The tool automatically generates a **blueprint-style infographic prompt** ready for AI image tools or pitch decks.

---

##  Features
- 📈 **Runway & Liftoff Simulation** – Compounds MRR and cost growth month-by-month  
- 💰 **Funding Gap + Recommended Raise** – Calculates true deficit to breakeven  
- ⚙️ **API-Ready** – Runs locally or as a FastAPI endpoint (`/runway`)  
- 🧠 **Prompt Generator** – Outputs clean, ready-to-copy image prompt text  
- 🪶 **Lightweight** – No external dependencies beyond Python standard library  

---

## 🧩 Example Output

🚀 RUN SUMMARY — Andiamo Tech

Cash: $2,000 | Burn: $140/mo → Static Runway: 14.3 mo

Growth: 10%/mo | Cost Growth: 2%/mo

Liftoff / Breakeven: 36 mo (2028-10-26)

Funding Gap: $2,667.89

❌ Requires $2,667.89 to reach liftoff. Recommended raise (+20% buffer): $3,202.

---

## 🧮 CLI Usage

Run interactively:
```bash
git clone https://github.com/andiamotech/RunwayToFlight.git
cd RunwayToFlight
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

python3 RunwayToFlight.py
```

🌐 Run as an API

FastAPI endpoint
```bash
uvicorn app:app --reload
```
Then open:
- Swagger UI: http://127.0.0.1:8000/docs
- POST Endpoint: /runway

🧠 How It Works

RunwayToFlight models your startup’s cash, growth, and burn dynamics:
- Projects revenue & cost growth monthly (compounded)
- Tracks cash depletion and auto-injects deficit funding
- Returns breakeven month, liftoff date, and cumulative deficit
- Suggests raise size, growth goal, or cost cut to reach liftoff

⚖️ License

Apache 2.0 © 2025 Andiamo Tech

https://andiamo.tech

Mobility with Meaning.
