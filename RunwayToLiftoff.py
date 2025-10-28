# runwaytoflight.py — RunwayToFlight v3.2 (FINAL)
# Founder Runway & Liftoff Planner
# Two clean sections:
#   1) Image-generation prompt (blueprint style) with ✅/❌ midline
#   2) Summary of job run (static runway + cumulative deficit)
# Features:
# - TRUE funding gap (sum of monthly shortfalls to liftoff)
# - Cumulative deficit = initial cash + funding injections − cash at breakeven
# - Recommended raise (+20% buffer) and alternative levers (growth% / cost cut%)
# - Inputs specify expected formats; no external deps.
# © 2025 Andiamo Tech | Apache-2.0

from pathlib import Path
import argparse, json, math
from datetime import date, datetime

DEFAULT_TARGET_RUNWAY_MONTHS = 24
DEFAULT_RAISE_BUFFER_PCT = 20.0
MAX_FUNDING_CAP = 10_000_000

# ----------------- Utils -----------------
def to_float(v):
    try:
        return float(str(v).replace(",", "").strip())
    except:
        return 0.0

def money(v):
    x = to_float(v)
    s = f"{x:,.2f}"
    return s[:-3] if s.endswith(".00") else s

def coerce_date(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                dt = datetime(dt.year, dt.month, 1)  # normalize to first of month
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    return s

def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
             31,30,31,30,31,31,30,31,30,31][m-1]
    return date(y, m, min(d.day, mdays))

def normalize_hex(c):
    c = (c or "").strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch*2 for ch in c)
    return f"#{c}" if c else "#12c04c"

# ----------------- Simulation -----------------
def simulate_to_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, max_months=240):
    """
    Month-by-month compounding of MRR and costs. If cash dips below 0,
    inject just-enough funding and accumulate as funding gap.
    Returns: (breakeven_month_index, breakeven_date, funding_gap, unreachable_flag, cash_end)
    """
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr = mrr0 if mrr0 > 0 else 1.0
    mrc = mrc0
    cash = cash0
    funding_gap = 0.0

    if mrr0 >= mrc0:
        # Already breakeven
        return 0, add_months(date.today(), 0).isoformat(), 0.0, False, cash

    for m in range(1, max_months + 1):
        mrr *= (1 + g)
        mrc *= (1 + cg)
        net = mrr - mrc
        cash += net
        if cash < 0:
            funding_gap += -cash
            cash = 0.0
        if funding_gap > MAX_FUNDING_CAP:
            return None, "N/A", float(MAX_FUNDING_CAP), True, cash
        if mrr >= mrc:
            return m, add_months(date.today(), m).isoformat(), round(funding_gap, 2), False, cash

    return None, "N/A", round(funding_gap, 2), True, cash

def simulate_survival_raise(mrr0, mrc0, g_pct, cg_pct, cash0, months):
    """Funding required to survive exactly `months` months (not necessarily to liftoff)."""
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr = mrr0 if mrr0 > 0 else 1.0
    mrc = mrc0
    cash = cash0
    f = 0.0
    for _ in range(months):
        mrr *= (1 + g)
        mrc *= (1 + cg)
        net = mrr - mrc
        cash += net
        if cash < 0:
            f += -cash
            cash = 0.0
        if f > MAX_FUNDING_CAP:
            return float(MAX_FUNDING_CAP), True
    return round(f, 2), False

# ----------------- Solvers (alternatives) -----------------
def solve_growth_for_zero_funding(mrr0, mrc0, cost_growth_pct, cash0, max_months=240):
    """Minimum monthly MRR growth % that yields zero funding gap, if feasible."""
    lo, hi = 0.0, 200.0
    if simulate_to_breakeven(mrr0, mrc0, hi, cost_growth_pct, cash0, max_months)[2] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        _, _, fund, _, _ = simulate_to_breakeven(mrr0, mrc0, mid, cost_growth_pct, cash0, max_months)
        if fund > 0:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)

def solve_cost_cut_for_zero_funding(mrr0, mrc0, growth_pct, cost_growth_pct, cash0, max_months=240):
    """Minimum % immediate cost cut (of current MRC) to yield zero funding gap given growth %."""
    lo, hi = 0.0, 0.9
    if simulate_to_breakeven(mrr0, mrc0*(1-hi), growth_pct, cost_growth_pct, cash0, max_months)[2] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        mrc_adj = mrc0 * (1 - mid)
        _, _, fund, _, _ = simulate_to_breakeven(mrr0, mrc_adj, growth_pct, cost_growth_pct, cash0, max_months)
        if fund > 0:
            lo = mid
        else:
            hi = mid
    return round(hi * 100, 1)

# ----------------- Compute -----------------
def compute(d):
    mrr = to_float(d["mrr"])
    last_mrr = to_float(d.get("last_mrr", mrr))
    mrc = to_float(d["mrc"])
    g = to_float(d.get("growth_pct", 0))
    cg = to_float(d.get("cost_growth_pct", 0))

    bootstrap = to_float(d.get("bootstrap_cash", 0))
    funding = to_float(d.get("external_equity_cash", 0))
    grant = to_float(d.get("grant_cash", 0))
    loan = to_float(d.get("loan_cash", 0))
    cash_pool = bootstrap + funding + grant + loan

    arr = mrr * 12
    net_burn = max(mrc - mrr, 0)
    runway_static = math.inf if net_burn == 0 else cash_pool / net_burn

    be_m, est_date, funding_gap, unreachable, cash_end = simulate_to_breakeven(
        mrr, mrc, g, cg, cash_pool, max_months=240
    )

    # Correct cumulative deficit accounting:
    # cumulative_deficit = initial cash + funding injections − cash at breakeven
    cumulative_deficit = (cash_pool + funding_gap - cash_end) if be_m is not None else None

    # Snapshot burn multiple now (if last_mrr > 0)
    net_new_arr_now = max((mrr - last_mrr) * 12, 0)
    burn_multiple_now = None if net_new_arr_now == 0 else (net_burn * 12) / net_new_arr_now

    survival_raise, _ = simulate_survival_raise(mrr, mrc, g, cg, cash_pool, DEFAULT_TARGET_RUNWAY_MONTHS)
    recommended_raise = math.ceil(max(funding_gap, survival_raise) * (1 + DEFAULT_RAISE_BUFFER_PCT/100.0))

    growth_to_zero = solve_growth_for_zero_funding(mrr, mrc, cg, cash_pool, 240)
    cut_to_zero = solve_cost_cut_for_zero_funding(mrr, mrc, g, cg, cash_pool, 240)

    return dict(
        arr=arr,
        net_burn=net_burn,
        runway_static=None if math.isinf(runway_static) else round(runway_static, 1),
        breakeven_months=be_m,
        est_date=est_date,
        funding_gap=funding_gap,
        cumulative_deficit=cumulative_deficit,
        unreachable=unreachable,
        cash_pool=cash_pool,
        growth_pct=g,
        cost_growth_pct=cg,
        bootstrap=bootstrap,
        funding=funding,
        survival_raise=survival_raise,
        recommended_raise=recommended_raise,
        growth_to_zero=growth_to_zero,
        cut_to_zero=cut_to_zero,
        burn_multiple=None if burn_multiple_now is None else round(burn_multiple_now, 2),
    )

# ----------------- Builders -----------------
def build_bottom_line(c):
    if c["funding_gap"] <= 0 and c["breakeven_months"] is not None:
        return f"✅ Liftoff in {c['breakeven_months']} mo ({c['est_date']}) without new funding."
    g_opt = f"{c['growth_to_zero']}%/mo" if c["growth_to_zero"] is not None else "N/A"
    c_opt = f"{c['cut_to_zero']}%" if c["cut_to_zero"] is not None else "N/A"
    return (f"❌ Requires ${money(c['funding_gap'])} to reach liftoff. "
            f"Recommended raise (+{int(DEFAULT_RAISE_BUFFER_PCT)}% buffer): ${money(c['recommended_raise'])}. "
            f"Alternatives → growth ≥ {g_opt} or cost cut ≥ {c_opt}%.")

def build_outputs(inp, c):
    accent = normalize_hex(inp["accent_colors"])
    company = inp["company_name"]
    liftoff = c["est_date"]

    # Midline message (one-sentence story)
    if c["funding_gap"] <= 0 and c["breakeven_months"] is not None:
        mid_text = f"✅ Liftoff in {c['breakeven_months']} mo ({liftoff}) without new funding."
    elif c["funding_gap"] > 0:
        mid_text = f"❌ Requires ${money(c['funding_gap'])} to reach liftoff."
    else:
        mid_text = "❌ Liftoff not in sight."

    # ---- SECTION 1: Image Prompt ----
    prompt = f"""COPY/PASTE INTO YOUR IMAGE GENERATOR — BEGIN
STYLE
Minimal, top-down vector infographic on a subtle blueprint grid.
Clean modern sans-serif font, blueprint blue background, white lines,
accent color {accent}. Aspect ratio: 16:9 (1792×1024).

SCENE COMPOSITION
- Horizontal runway centered across the frame.
- Sleek futuristic aircraft labeled “{company}” mid-runway.
- Left label: “Formation {inp['formation_date']}”.
- Right label: “Est. Liftoff {liftoff}”.
- Four small data tags near runway:
    • Burn ${money(c['net_burn'])}/mo
    • Capital ${money(c['cash_pool'])}
    • Liftoff {liftoff}
    • Funding Gap ${money(c['funding_gap'])}

MIDLINE TEXT
“{mid_text}”
COPY/PASTE INTO YOUR IMAGE GENERATOR — END""".strip()

    # ---- SECTION 2: Summary ----
    run = "∞ (cash covers burn)" if c["runway_static"] is None else f"{c['runway_static']} mo"
    bm_now = "n/a" if c["burn_multiple"] is None else f"{c['burn_multiple']}x"

    cumul_lines = ""
    if c["cumulative_deficit"] is not None:
        cumul_lines = (
            f"\n📊 Financial Snapshot"
            f"\nCumulative Deficit to Liftoff: ${money(c['cumulative_deficit'])}"
            f"\nTotal Cash Deployed: ${money(c['cash_pool'])}"
            f"\nUnfunded Gap: ${money(c['funding_gap'])}"
        )

    summary = f"""🚀 RUN SUMMARY — {company}
────────────────────────────────────────────
Cash Available: ${money(c['cash_pool'])}
Monthly Burn: ${money(c['net_burn'])} → Static Runway: {run}

Growth: +{c['growth_pct']}%/mo   |   Cost Growth: +{c['cost_growth_pct']}%/mo
Liftoff / Breakeven: {('N/A' if c['breakeven_months'] is None else f"{c['breakeven_months']} mo ({c['est_date']})")}
ARR: ${money(c['arr'])}   |   Burn Multiple: {bm_now}
{cumul_lines}

────────────────────────────────────────────
{build_bottom_line(c)}""".strip()

    return prompt, summary

# ----------------- Inputs -----------------
def load_inputs():
    print("\n🛫 RunwayToFlight — Founder Runway & Liftoff Planner (12 steps)\n")
    ask = lambda i, t: input(f"[{i}/12] {t}: ").strip()
    d = {}
    print("\n🏢 SECTION — Company")
    d["company_name"]   = ask(1, "Company name (text, e.g. Andiamo Tech)")
    d["formation_date"] = coerce_date(ask(2, "Formation date (YYYY-MM or YYYY-MM-DD)"))

    print("\n💵 SECTION — Revenue & Costs")
    d["mrr"]      = ask(3, "Current MRR (number, USD/month)")
    d["last_mrr"] = ask(4, "Last month MRR (number, USD/month)")
    d["mrc"]      = ask(5, "Total monthly costs MRC (number, USD/month)")

    print("\n📈 SECTION — Trajectory")
    d["growth_pct"]      = ask(6, "Expected MRR growth % per month (number, e.g. 10)")
    d["cost_growth_pct"] = ask(7, "Cost growth % per month (number, e.g. 2)")

    print("\n🏗️ SECTION — Capital Sources")
    d["bootstrap_cash"]       = ask(8, "Bootstrap / founder funds (number, USD)")
    d["external_equity_cash"] = ask(9, "Angel+VC funding combined (number, USD)")
    d["grant_cash"]           = ask(10, "Grants / non-dilutive funds (number, USD)")
    d["loan_cash"]            = ask(11, "Loans or credit lines (number, USD)")

    print("\n🎨 SECTION — Visual")
    d["accent_colors"] = ask(12, "Accent colors (hex, e.g. #12c04c)")
    return d

# ----------------- File I/O -----------------
def save_files(prompt, summary, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt)
    (out / "summary.txt").write_text(summary)
    print(f"\n✅ Files saved to {out.resolve()}")

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(description="RunwayToFlight — Founder Runway & Liftoff Planner")
    parser.add_argument("--json", help="Optional JSON inputs")
    parser.add_argument("--outdir", default="runwaytoflight_out")
    args = parser.parse_args()

    if args.json:
        inputs = json.loads(args.json)
        if "formation_date" in inputs:
            inputs["formation_date"] = coerce_date(inputs["formation_date"])
        if "accent_colors" not in inputs:
            inputs["accent_colors"] = "#12c04c"
    else:
        inputs = load_inputs()

    calc = compute(inputs)
    prompt, summary = build_outputs(inputs, calc)
    save_files(prompt, summary, args.outdir)

    print("\n=== PROMPT ===\n", prompt)
    print("\n=== SUMMARY ===\n", summary)

if __name__ == "__main__":
    main()