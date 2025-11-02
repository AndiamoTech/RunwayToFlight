# runwaytoflight.py — RunwayToFlight v3.3
# Founder Runway & Liftoff Planner
# - Debt-aware (3% APR, 36 mo amortization)
# - Operational liftoff requires covering debt while loan active
# - Debt-aware static runway
# - Accurate "serviceable through revenue" indicator
# - Funding cap messaging, strict date fallback, “15%” inputs
from pathlib import Path
import argparse, json, math, re
from datetime import date, datetime

DEFAULT_TARGET_RUNWAY_MONTHS = 24
DEFAULT_RAISE_BUFFER_PCT = 20.0
MAX_FUNDING_CAP = 10_000_000

# Debt defaults
LOAN_TERM_MONTHS = 36
LOAN_ANNUAL_RATE = 0.03  # 3% APR

# ----------------- Utils -----------------
def to_float(v):
    try:
        s = str(v).strip().replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        return float(s)
    except:
        return 0.0

def money(v):
    x = to_float(v)
    s = f"{x:,.2f}"
    return s[:-3] if s.endswith(".00") else s

def coerce_date(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d","%Y-%m"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                dt = datetime(dt.year, dt.month, 1)
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    # Fallback: first of current month
    dt = datetime(date.today().year, date.today().month, 1)
    return dt.strftime("%Y-%m-%d")

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

def _strip_numeric_commas(s: str) -> str:
    return re.sub(r'(?<=\d),(?=\d{3}(\D|$))', '', s)

def loan_monthly_payment(principal: float, term_mo: int, apr: float) -> float:
    if principal <= 0 or term_mo <= 0:
        return 0.0
    r = apr / 12.0
    if r == 0:
        return principal / term_mo
    return principal * (r / (1 - (1 + r) ** (-term_mo)))

# ----------------- Simulation -----------------
def simulate_operational_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, loan_principal, max_months=240):
    """
    Operational liftoff condition:
      - While loan active (month <= LOAN_TERM_MONTHS): MRR >= MRC + monthly_debt
      - After loan term: MRR >= MRC
    """
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr, mrc, cash, funding_gap = mrr0, mrc0, cash0, 0.0
    rem_principal = loan_principal
    pmt = loan_monthly_payment(loan_principal, LOAN_TERM_MONTHS, LOAN_ANNUAL_RATE)

    # Immediate liftoff check with debt condition
    need_debt_cover = (1 <= LOAN_TERM_MONTHS and pmt > 0)
    if (mrr >= mrc + (pmt if need_debt_cover else 0)):
        return dict(month=0, date=add_months(date.today(), 0).isoformat(),
                    funding_gap=0.0, unreachable=False, cash=cash,
                    loan_remaining=round(rem_principal, 2),
                    mrr_at_op=mrr, mrc_at_op=mrc, pmt_at_op=(pmt if need_debt_cover else 0))

    for m in range(1, max_months + 1):
        mrr *= (1 + g); mrc *= (1 + cg)
        net = mrr - mrc
        cash += net

        # Loan service
        if m <= LOAN_TERM_MONTHS and rem_principal > 0 and pmt > 0:
            interest = rem_principal * (LOAN_ANNUAL_RATE / 12.0)
            principal_paid = max(pmt - interest, 0)
            rem_principal = max(rem_principal - principal_paid, 0)
            cash -= pmt

        if cash < 0:
            funding_gap += -cash
            cash = 0.0
        if funding_gap > MAX_FUNDING_CAP:
            break

        need_debt_cover = (m <= LOAN_TERM_MONTHS and pmt > 0)
        if mrr >= mrc + (pmt if need_debt_cover else 0):
            return dict(month=m, date=add_months(date.today(), m).isoformat(),
                        funding_gap=round(funding_gap, 2), unreachable=False, cash=cash,
                        loan_remaining=round(rem_principal, 2),
                        mrr_at_op=mrr, mrc_at_op=mrc, pmt_at_op=(pmt if need_debt_cover else 0))

    return dict(month=None, date="N/A",
                funding_gap=min(funding_gap, float(MAX_FUNDING_CAP)),
                unreachable=True, cash=cash, loan_remaining=round(rem_principal, 2),
                mrr_at_op=None, mrc_at_op=None, pmt_at_op=None)

def simulate_cash_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, loan_principal, max_months=240):
    """
    Cash BE requires:
      1) operational BE (with debt condition) reached, and
      2) cash >= starting cash
    """
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr, mrc, cash, funding_gap = mrr0, mrc0, cash0, 0.0
    op_month, start_cash_level = None, cash0
    rem_principal = loan_principal
    pmt = loan_monthly_payment(loan_principal, LOAN_TERM_MONTHS, LOAN_ANNUAL_RATE)

    for m in range(1, max_months + 1):
        mrr *= (1 + g); mrc *= (1 + cg)
        net = mrr - mrc
        cash += net

        # Loan service
        if m <= LOAN_TERM_MONTHS and rem_principal > 0 and pmt > 0:
            interest = rem_principal * (LOAN_ANNUAL_RATE / 12.0)
            principal_paid = max(pmt - interest, 0)
            rem_principal = max(rem_principal - principal_paid, 0)
            cash -= pmt

        if cash < 0:
            funding_gap += -cash
            cash = 0.0
        if funding_gap > MAX_FUNDING_CAP:
            break

        need_debt_cover = (m <= LOAN_TERM_MONTHS and pmt > 0)
        if op_month is None and mrr >= mrc + (pmt if need_debt_cover else 0):
            op_month = m

        if op_month is not None and cash >= start_cash_level:
            return dict(op_month=op_month, cash_month=m,
                        date=add_months(date.today(), m).isoformat(),
                        funding_gap=round(funding_gap, 2),
                        unreachable=False, cash=cash,
                        loan_remaining=round(rem_principal, 2))

    return dict(op_month=op_month, cash_month=None, date="N/A",
                funding_gap=min(funding_gap, float(MAX_FUNDING_CAP)),
                unreachable=True, cash=cash, loan_remaining=round(rem_principal, 2))

def simulate_survival_raise(mrr0, mrc0, g_pct, cg_pct, cash0, loan_principal, months):
    g, cg = g_pct / 100.0, cg_pct / 100.0
    mrr, mrc, cash, f = mrr0, mrc0, cash0, 0.0
    rem_principal = loan_principal
    pmt = loan_monthly_payment(loan_principal, LOAN_TERM_MONTHS, LOAN_ANNUAL_RATE)

    for m in range(1, months + 1):
        mrr *= (1 + g); mrc *= (1 + cg)
        net = mrr - mrc
        cash += net

        if m <= LOAN_TERM_MONTHS and rem_principal > 0 and pmt > 0:
            interest = rem_principal * (LOAN_ANNUAL_RATE / 12.0)
            principal_paid = max(pmt - interest, 0)
            rem_principal = max(rem_principal - principal_paid, 0)
            cash -= pmt

        if cash < 0:
            f += -cash
            cash = 0.0
        if f > MAX_FUNDING_CAP:
            return float(MAX_FUNDING_CAP), True
    return round(f, 2), False

# ----------------- Solvers -----------------
def simulate_op_for_solver(mrr0, mrc0, g_pct, cg_pct, cash0, loan_principal, max_months=240):
    return simulate_operational_breakeven(mrr0, mrc0, g_pct, cg_pct, cash0, loan_principal, max_months)

def solve_growth_for_zero_funding(mrr0, mrc0, cost_growth_pct, cash0, loan_principal, max_months=240):
    lo, hi = 0.0, 200.0
    if simulate_op_for_solver(mrr0, mrc0, hi, cost_growth_pct, cash0, loan_principal, max_months)['funding_gap'] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        fund = simulate_op_for_solver(mrr0, mrc0, mid, cost_growth_pct, cash0, loan_principal, max_months)['funding_gap']
        if fund > 0: lo = mid
        else: hi = mid
    return round(hi, 2)

def solve_cost_cut_for_zero_funding(mrr0, mrc0, growth_pct, cost_growth_pct, cash0, loan_principal, max_months=240):
    lo, hi = 0.0, 0.9
    if simulate_op_for_solver(mrr0, mrc0*(1-hi), growth_pct, cost_growth_pct, cash0, loan_principal, max_months)['funding_gap'] > 0:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        mrc_adj = mrc0 * (1 - mid)
        fund = simulate_op_for_solver(mrr0, mrc_adj, growth_pct, cost_growth_pct, cash0, loan_principal, max_months)['funding_gap']
        if fund > 0: lo = mid
        else: hi = mid
    return round(hi * 100, 1)

# ----------------- Compute -----------------
def compute(d):
    mrr = to_float(d["mrr"]); last_mrr = to_float(d.get("last_mrr", mrr))
    mrc = to_float(d["mrc"]); g = to_float(d.get("growth_pct", 0))
    cg = to_float(d.get("cost_growth_pct", 0))

    bootstrap = to_float(d.get("bootstrap_cash", 0))
    funding   = to_float(d.get("external_equity_cash", 0))
    grant     = to_float(d.get("grant_cash", 0))
    loan      = to_float(d.get("loan_cash", 0))
    cash_pool = bootstrap + funding + grant + loan

    auto_seeded = False
    if mrr == 0 and last_mrr == 0:
        mrr = 1.0; auto_seeded = True

    arr = mrr * 12
    # Debt-aware static runway
    monthly_debt = loan_monthly_payment(loan, LOAN_TERM_MONTHS, LOAN_ANNUAL_RATE)
    net_burn_static = max(mrc + (monthly_debt if loan > 0 else 0) - mrr, 0)
    runway_static = math.inf if net_burn_static == 0 else cash_pool / net_burn_static

    op_res   = simulate_operational_breakeven(mrr, mrc, g, cg, cash_pool, loan, 240)
    cash_res = simulate_cash_breakeven(mrr, mrc, g, cg, cash_pool, loan, 240)

    cumulative_deficit_op   = (cash_pool + op_res['funding_gap'] - op_res['cash'])   if op_res['month'] is not None else None
    cumulative_deficit_cash = (cash_pool + cash_res['funding_gap'] - cash_res['cash']) if cash_res['cash_month'] is not None else None

    net_new_arr_now = max((mrr - last_mrr) * 12, 0)
    burn_multiple_now = None if (net_new_arr_now == 0 or auto_seeded) else ( (mrc - mrr) * 12 ) / net_new_arr_now if (mrc - mrr) > 0 else 0.0

    survival_raise, _ = simulate_survival_raise(mrr, mrc, g, cg, cash_pool, loan, DEFAULT_TARGET_RUNWAY_MONTHS)
    funding_gap_target = op_res['funding_gap'] if op_res['month'] is not None else cash_res['funding_gap']
    recommended_raise = math.ceil(max(funding_gap_target, survival_raise) * (1 + DEFAULT_RAISE_BUFFER_PCT/100.0))

    growth_to_zero = solve_growth_for_zero_funding(mrr, mrc, cg, cash_pool, loan, 240)
    cut_to_zero    = solve_cost_cut_for_zero_funding(mrr, mrc, g, cg, cash_pool, loan, 240)

    cap_reached = (op_res['funding_gap'] >= float(MAX_FUNDING_CAP)) or (cash_res['funding_gap'] >= float(MAX_FUNDING_CAP))

    # Indicators at operational BE
    serviceable = None
    if op_res['month'] is not None:
        mrr_at_op = op_res.get('mrr_at_op') or 0
        mrc_at_op = op_res.get('mrc_at_op') or 0
        pmt_at_op = op_res.get('pmt_at_op') or 0
        net_after_debt = mrr_at_op - mrc_at_op - pmt_at_op
        serviceable = (net_after_debt >= 0)
    else:
        mrr_at_op = mrc_at_op = pmt_at_op = None

    return dict(
        arr=arr,
        net_burn=max(mrc - mrr, 0),
        runway_static=None if math.isinf(runway_static) else round(runway_static, 1),
        op_breakeven_months=op_res['month'],  op_breakeven_date=op_res['date'],
        cash_breakeven_months=cash_res['cash_month'], cash_breakeven_date=cash_res['date'],
        op_funding_gap=op_res['funding_gap'], cash_funding_gap=cash_res['funding_gap'],
        cumulative_deficit_op=cumulative_deficit_op, cumulative_deficit_cash=cumulative_deficit_cash,
        unreachable=op_res['unreachable'] and cash_res['unreachable'],
        cash_pool=cash_pool, growth_pct=g, cost_growth_pct=cg,
        bootstrap=bootstrap, funding=funding, grant=grant, loan_cash=loan,
        survival_raise=survival_raise, recommended_raise=recommended_raise,
        growth_to_zero=growth_to_zero, cut_to_zero=cut_to_zero,
        burn_multiple=None if burn_multiple_now is None else round(burn_multiple_now, 2),
        auto_seeded=auto_seeded,
        loan_term_months=LOAN_TERM_MONTHS, loan_apr=LOAN_ANNUAL_RATE, loan_monthly_payment=monthly_debt,
        loan_remaining_at_op=op_res.get('loan_remaining'), loan_remaining_at_cash=cash_res.get('loan_remaining'),
        cap_reached=cap_reached,
        mrr_at_op=mrr_at_op, mrc_at_op=mrc_at_op, pmt_at_op=pmt_at_op,
        serviceable_at_op=serviceable
    )

# ----------------- Builders -----------------
def build_bottom_line(c):
    if c["op_funding_gap"] <= 0 and c["op_breakeven_months"] is not None:
        return f"✅ Operational liftoff in {c['op_breakeven_months']} mo ({c['op_breakeven_date']}) without new funding."

    g_opt = f"{c['growth_to_zero']}% per month" if c["growth_to_zero"] is not None else "N A"
    c_opt = f"{c['cut_to_zero']}%" if c["cut_to_zero"] is not None else "N A"

    if c.get("cap_reached"):
        return ("❌ Model hit the funding cap of $10,000,000. Current assumptions are unlikely to reach liftoff. "
                f"Try faster revenue growth {g_opt} or a deeper cost cut {c_opt}, then rerun.")

    return (f"❌ Requires ${money(c['op_funding_gap'])} to reach operational liftoff. "
            f"Recommended raise plus {int(DEFAULT_RAISE_BUFFER_PCT)} percent buffer: ${money(c['recommended_raise'])}. "
            f"Alternatives → growth at least {g_opt} or cost cut at least {c_opt}.")

def build_outputs(inp, c):
    accent = normalize_hex(inp["accent_colors"])
    company = inp.get("company_name") or "Andiamo Tech"
    liftoff = c["op_breakeven_date"] if c["op_breakeven_date"] != "N/A" else "N/A"

    if c.get("cap_reached"):
        mid_text = "❌ Model hit cap at $10,000,000. Adjust assumptions and rerun."
    else:
        if c["op_funding_gap"] <= 0 and c["op_breakeven_months"] is not None:
            mid_text = f"✅ Liftoff in {c['op_breakeven_months']} mo ({liftoff}) without new funding."
        elif c["op_funding_gap"] > 0:
            mid_text = f"❌ Requires ${money(c['op_funding_gap'])} to reach liftoff."
        else:
            mid_text = "❌ Liftoff not in sight."

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

    run = "∞ (cash covers burn)" if c["runway_static"] is None else f"{c['runway_static']} mo"
    bm_now = "n/a" if c["burn_multiple"] is None else f"{c['burn_multiple']}x"

    op_text   = "N/A" if c["op_breakeven_months"]   is None else f"{c['op_breakeven_months']} mo ({c['op_breakeven_date']})"
    cash_text = "N/A" if c["cash_breakeven_months"] is None else f"{c['cash_breakeven_months']} mo ({c['cash_breakeven_date']})"

    cash_block = (
        f"• Available: ${money(c['cash_pool'])}\n"
        f"• Monthly Burn: ${money(c['net_burn'])}\n"
        f"• Static Runway: {run}\n"
        f"• Cash Sources → Bootstrap ${money(c['bootstrap'])}, Equity ${money(c['funding'])}, Grants ${money(c['grant'])}, Loan ${money(c['loan_cash'])}"
    )

    serv = c.get("serviceable_at_op")
    serviceable_str = "✅ Yes" if serv is True else ("⚠️ No" if serv is False else "N/A")
    key_indicators = [
        f"• Loan serviceable through revenue by breakeven: {serviceable_str}",
        "• Loan fully repaid before cash runway exhausted: " +
        ("✅ Yes" if (c.get('loan_remaining_at_cash', 0) == 0) else "⚠️ No"),
        f"• Loan Remaining → Operational BE: ${money(c.get('loan_remaining_at_op', 0))} | Cash BE: ${money(c.get('loan_remaining_at_cash', 0))}"
    ]

    growth_section = f"""📊 GROWTH & COST TRAJECTORY
• Revenue Growth: +{c['growth_pct']}% per month
• Cost Growth: +{c['cost_growth_pct']}% per month

💡 Key Indicators
{chr(10).join(key_indicators)}"""

    milestones_section = f"""📈 MILESTONES
• Operational Breakeven: {op_text}
• Cash Breakeven: {cash_text}
• Cash Used by Operational BE: ${money(abs(c.get('cumulative_deficit_op') or 0))}
• Debt Service: ${money(c['loan_monthly_payment'])}/mo × {c['loan_term_months']} mo @ {round(c['loan_apr']*100,2)}% APR"""

    summary = f"""🚀 RUN SUMMARY — {company}
────────────────────────────────────────────
💰 CASH POSITION
{cash_block}

{growth_section}

{milestones_section}

🏎️ EFFICIENCY METRICS
• ARR: {"Pre revenue (seeded $1)" if c["auto_seeded"] else "$"+money(c["arr"])}
• Burn Multiple: {bm_now}

────────────────────────────────────────────
{build_bottom_line(c)}""".strip()

    if c.get("cap_reached"):
        summary += ("\n\n⚠️ NOTE\n"
                    "• Model reached the hard cap of ten million dollars in funding gap. "
                    "Consider higher revenue growth, lower costs, or a smaller scope and rerun.")

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
    d["mrr"]      = ask(3, "Current MRR (number, USD per month)")
    d["last_mrr"] = ask(4, "Last month MRR (number, USD per month)")
    d["mrc"]      = ask(5, "Total monthly costs MRC (number, USD per month)")

    print("\n📈 SECTION — Trajectory")
    d["growth_pct"]      = ask(6, "Expected MRR growth percent per month (number, e.g. 10 or 10%)")
    d["cost_growth_pct"] = ask(7, "Cost growth percent per month (number, e.g. 2 or 2%)")

    print("\n🏗️ SECTION — Capital Sources")
    d["bootstrap_cash"]       = ask(8,  "Bootstrap or founder funds (number, USD)")
    d["external_equity_cash"] = ask(9,  "Angel and VC funding combined (number, USD)")
    d["grant_cash"]           = ask(10, "Grants or non dilutive funds (number, USD)")
    d["loan_cash"]            = ask(11, "Loans or credit lines (number, USD)")

    print("\n🎨 SECTION — Visual")
    d["accent_colors"] = "#12c04c"
    return d

# ----------------- File I/O -----------------
def save_files(prompt, summary, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
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
        raw = args.json
        try:
            inputs = json.loads(raw)
        except json.JSONDecodeError:
            inputs = json.loads(_strip_numeric_commas(raw))
        if "formation_date" in inputs:
            inputs["formation_date"] = coerce_date(inputs["formation_date"])
        inputs["accent_colors"] = inputs.get("accent_colors", "#12c04c")
    else:
        inputs = load_inputs()

    calc = compute(inputs)
    prompt, summary = build_outputs(inputs, calc)
    save_files(prompt, summary, args.outdir)
    print("\n=== PROMPT ===\n", prompt)
    print("\n=== SUMMARY ===\n", summary)

if __name__ == "__main__":
    main()
