# runwaytoflight.py — RunwayToFlight v3.5 (AUTO-SEED + CLEAR SUMMARY)
# Founder Runway & Liftoff Planner
# Sections:
#   1) Image-generation prompt (blueprint style) with ✅/❌ midline
#   2) Summary of job run (clear, investor-friendly)
# Features:
# - Operational and Cash Breakeven (separate + explicit)
# - TRUE funding gap (sum of monthly shortfalls to breakeven)
# - Cumulative deficit = initial cash + funding injections − cash at breakeven
# - Recommended raise (+20% buffer) and alternative levers (growth% / cost cut%)
# - Auto-seeds MRR to $1 when both current and last-month MRR are 0 (no prompt)
# - Cleaner summary with “Net Cash Remaining/Surplus at Cash Breakeven”
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
                dt = datetime(dt.year, dt.month, 1)
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
def simulate_operational_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, max_months=240):
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr = mrr0
    mrc = mrc0
    cash = cash0
    funding_gap = 0.0

    if mrr >= mrc:
        return dict(month=0, date=add_months(date.today(), 0).isoformat(),
                    funding_gap=0.0, unreachable=False, cash=cash)

    for m in range(1, max_months + 1):
        mrr *= (1 + g)
        mrc *= (1 + cg)
        net = mrr - mrc
        cash += net
        if cash < 0:
            funding_gap += -cash
            cash = 0.0
        if mrr >= mrc:
            return dict(month=m, date=add_months(date.today(), m).isoformat(),
                        funding_gap=round(funding_gap, 2), unreachable=False, cash=cash)
        if funding_gap > MAX_FUNDING_CAP:
            break

    return dict(month=None, date="N/A",
                funding_gap=min(funding_gap, float(MAX_FUNDING_CAP)),
                unreachable=True, cash=cash)

def simulate_cash_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, max_months=240):
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr = mrr0
    mrc = mrc0
    cash = cash0
    funding_gap = 0.0
    op_month = None
    start_cash_level = cash0

    for m in range(1, max_months + 1):
        mrr *= (1 + g)
        mrc *= (1 + cg)
        net = mrr - mrc
        cash += net
        if cash < 0:
            funding_gap += -cash
            cash = 0.0

        if op_month is None and mrr >= mrc:
            op_month = m

        if op_month is not None and funding_gap == 0.0 and cash >= start_cash_level:
            return dict(op_month=op_month, cash_month=m,
                        date=add_months(date.today(), m).isoformat(),
                        funding_gap=round(funding_gap, 2),
                        unreachable=False, cash=cash)

        if funding_gap > MAX_FUNDING_CAP:
            break

    return dict(op_month=op_month, cash_month=None, date="N/A",
                funding_gap=min(funding_gap, float(MAX_FUNDING_CAP)),
                unreachable=True, cash=cash)

def simulate_survival_raise(mrr0, mrc0, g_pct, cg_pct, cash0, months):
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr = mrr0
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

# ----------------- Solvers -----------------
def solve_growth_for_zero_funding(mrr0, mrc0, cost_growth_pct, cash0, max_months=240):
    lo, hi = 0.0, 200.0
    if simulate_operational_breakeven(mrr0, mrc0, hi, cost_growth_pct, cash0, max_months)['funding_gap'] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        fund = simulate_operational_breakeven(mrr0, mrc0, mid, cost_growth_pct, cash0, max_months)['funding_gap']
        if fund > 0:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)

def solve_cost_cut_for_zero_funding(mrr0, mrc0, growth_pct, cost_growth_pct, cash0, max_months=240):
    lo, hi = 0.0, 0.9
    if simulate_operational_breakeven(mrr0, mrc0*(1-hi), growth_pct, cost_growth_pct, cash0, max_months)['funding_gap'] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        mrc_adj = mrc0 * (1 - mid)
        fund = simulate_operational_breakeven(mrr0, mrc_adj, growth_pct, cost_growth_pct, cash0, max_months)['funding_gap']
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

    # Auto-seed if pre-revenue with no prior MRR
    auto_seeded = False
    if mrr == 0 and last_mrr == 0:
        mrr = 1.0
        auto_seeded = True

    arr = mrr * 12
    net_burn = max(mrc - mrr, 0)
    runway_static = math.inf if net_burn == 0 else cash_pool / net_burn

    op_res = simulate_operational_breakeven(mrr, mrc, g, cg, cash_pool, max_months=240)
    cash_res = simulate_cash_breakeven(mrr, mrc, g, cg, cash_pool, max_months=240)

    cumulative_deficit_op = (cash_pool + op_res['funding_gap'] - op_res['cash']) if op_res['month'] is not None else None
    cumulative_deficit_cash = (cash_pool + cash_res['funding_gap'] - cash_res['cash']) if cash_res['cash_month'] is not None else None

    # Burn multiple (skip if auto-seeded to avoid misleading metric)
    net_new_arr_now = max((mrr - last_mrr) * 12, 0)
    burn_multiple_now = None if (net_new_arr_now == 0 or auto_seeded) else (net_burn * 12) / net_new_arr_now

    # Funding to survive target months vs to reach op/cash BE
    survival_raise, _ = simulate_survival_raise(mrr, mrc, g, cg, cash_pool, DEFAULT_TARGET_RUNWAY_MONTHS)
    funding_gap_target = op_res['funding_gap'] if op_res['month'] is not None else cash_res['funding_gap']
    recommended_raise = math.ceil(max(funding_gap_target, survival_raise) * (1 + DEFAULT_RAISE_BUFFER_PCT/100.0))

    growth_to_zero = solve_growth_for_zero_funding(mrr, mrc, cg, cash_pool, 240)
    cut_to_zero = solve_cost_cut_for_zero_funding(mrr, mrc, g, cg, cash_pool, 240)

    # Net cash remaining/surplus at cash breakeven
    net_cash_remaining = None
    net_cash_surplus = None
    if cash_res['cash_month'] is not None:
        delta = cash_res['cash'] - cash_pool
        if delta >= 0:
            net_cash_surplus = round(delta, 2)  # how much above starting cash
        else:
            # if condition was met by equality, this shouldn't happen, but guard anyway
            net_cash_remaining = round(cash_res['cash'], 2)

    return dict(
        arr=arr,
        net_burn=net_burn,
        runway_static=None if math.isinf(runway_static) else round(runway_static, 1),
        op_breakeven_months=op_res['month'],
        op_breakeven_date=op_res['date'],
        cash_breakeven_months=cash_res['cash_month'],
        cash_breakeven_date=cash_res['date'],
        op_funding_gap=op_res['funding_gap'],
        cash_funding_gap=cash_res['funding_gap'],
        cumulative_deficit_op=cumulative_deficit_op,
        cumulative_deficit_cash=cumulative_deficit_cash,
        net_cash_remaining_at_cash_be=net_cash_remaining,
        net_cash_surplus_at_cash_be=net_cash_surplus,
        unreachable=op_res['unreachable'] and cash_res['unreachable'],
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
        auto_seeded=auto_seeded,
    )

# ----------------- Builders -----------------
def build_bottom_line(c):
    if c["op_funding_gap"] <= 0 and c["op_breakeven_months"] is not None:
        return f"✅ Operational liftoff in {c['op_breakeven_months']} mo ({c['op_breakeven_date']}) without new funding."
    g_opt = f"{c['growth_to_zero']}%/mo" if c["growth_to_zero"] is not None else "N/A"
    c_opt = f"{c['cut_to_zero']}%" if c["cut_to_zero"] is not None else "N/A"
    return (f"❌ Requires ${money(c['op_funding_gap'])} to reach **operational** liftoff. "
            f"Recommended raise (+{int(DEFAULT_RAISE_BUFFER_PCT)}% buffer): ${money(c['recommended_raise'])}. "
            f"Alternatives → growth ≥ {g_opt} or cost cut ≥ {c_opt}%.")

def build_outputs(inp, c):
    accent = normalize_hex(inp["accent_colors"])
    company = inp["company_name"] or "Andiamo Tech"
    liftoff = c["op_breakeven_date"] if c["op_breakeven_date"] != "N/A" else "N/A"

    # Midline message
    if c["op_funding_gap"] <= 0 and c["op_breakeven_months"] is not None:
        mid_text = f"✅ Liftoff in {c['op_breakeven_months']} mo ({liftoff}) without new funding."
    elif c["op_funding_gap"] > 0:
        mid_text = f"❌ Requires ${money(c['op_funding_gap'])} to reach liftoff."
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
    • Funding Gap ${money(c['op_funding_gap'])}

MIDLINE TEXT
“{mid_text}”
COPY/PASTE INTO YOUR IMAGE GENERATOR — END""".strip()

    # ---- SECTION 2: Summary ----
    run = "∞ (cash covers burn)" if c["runway_static"] is None else f"{c['runway_static']} mo"
    bm_now = "n/a" if c["burn_multiple"] is None else f"{c['burn_multiple']}x"

    op_text = "N/A" if c["op_breakeven_months"] is None else f"{c['op_breakeven_months']} mo ({c['op_breakeven_date']})"
    cash_text = "N/A" if c["cash_breakeven_months"] is None else f"{c['cash_breakeven_months']} mo ({c['cash_breakeven_date']})"

    # Build milestone extras
    extra_lines = [
        f"Operational Breakeven: {op_text}",
        f"Cash Breakeven: {cash_text}",
        f"Cash Used by Operational Breakeven: ${money(abs(c.get('cumulative_deficit_op') or 0))}",
    ]
    # Cash breakeven cash position
    if c["cash_breakeven_months"] is not None:
        if c["net_cash_surplus_at_cash_be"] is not None:
            extra_lines.append(f"Net Cash Surplus at Cash Breakeven: ${money(c['net_cash_surplus_at_cash_be'])}")
        elif c["net_cash_remaining_at_cash_be"] is not None:
            extra_lines.append(f"Net Cash Remaining at Cash Breakeven: ${money(c['net_cash_remaining_at_cash_be'])}")
        else:
            # exactly at starting cash
            extra_lines.append("Net Cash at Cash Breakeven: $0 (back to starting level)")

    if c["auto_seeded"]:
        extra_lines.append("ℹ️ Auto-seeded with $1 MRR to simulate first customer (pre-revenue).")

    extra = "\n".join(extra_lines)

    summary = f"""🚀 RUN SUMMARY — {company}
────────────────────────────────────────────
💰 CASH POSITION
• Available: ${money(c['cash_pool'])}
• Monthly Burn: ${money(c['net_burn'])}
• Static Runway: {run}

📊 GROWTH & COST TRAJECTORY
• Revenue Growth: +{c['growth_pct']}%/mo
• Cost Growth: +{c['cost_growth_pct']}%/mo

📈 MILESTONES
{extra}

🏎️ EFFICIENCY METRICS
• ARR: ${money(c['arr'])}
• Burn Multiple: {bm_now}

────────────────────────────────────────────
{build_bottom_line(c)}""".strip()

    return prompt, summary

# ----------------- Inputs -----------------
def load_inputs():
    print("\n🛫 RunwayToFlight — Founder Runway & Liftoff Planner (11 steps)\n")
    ask = lambda i, t: input(f"[{i}/11] {t}: ").strip()
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
    d["bootstrap_cash"]       = ask(8,  "Bootstrap / founder funds (number, USD)")
    d["external_equity_cash"] = ask(9,  "Angel+VC funding combined (number, USD)")
    d["grant_cash"]           = ask(10, "Grants / non-dilutive funds (number, USD)")
    d["loan_cash"]            = ask(11, "Loans or credit lines (number, USD)")

    print("\n🎨 SECTION — Visual")
    d["accent_colors"] = "#12c04c"
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
