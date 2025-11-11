#!/usr/bin/env python3
# runwaytoflight.py — RunwayToFlight v3.4
# Founder Runway & Liftoff Planner (debt-aware)
# © 2025 Andiamo Tech | Apache-2.0

from pathlib import Path
import argparse, json, math
from datetime import date, datetime

# ---------- Constants ----------
DEFAULT_TARGET_RUNWAY_MONTHS = 24
DEFAULT_RAISE_BUFFER_PCT = 20.0
MAX_FUNDING_CAP = 10_000_000
EPS = 1e-6

# ---------- Utils ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_float(v):
    try:
        return float(str(v).replace(",", "").strip())
    except:
        return 0.0

def money(v, symbol="$"):
    x = to_float(v)
    s = f"{x:,.2f}"
    return f"{symbol}{s}"

def strip_money(v):
    return to_float(str(v).replace("$",""))

def coerce_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    return s

def parse_base_date(s):
    try:
        return datetime.strptime((s or "").strip(), "%Y-%m-%d").date()
    except:
        return date.today()

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

def em(flag, no_emoji=False):
    return "Yes" if flag and no_emoji else ("No" if no_emoji else ("✅" if flag else "⚠️"))

# ---------- Loan math ----------
def loan_monthly_payment(principal, apr_dec, term_mo):
    if principal <= EPS or term_mo <= 0:
        return 0.0
    r = apr_dec/12.0
    if abs(r) < EPS:
        return principal / term_mo
    return principal * (r * (1+r)**term_mo) / ((1+r)**term_mo - 1)

def step_loan(rem, apr_dec, pmt):
    if rem <= EPS or pmt <= 0:
        return rem, 0.0, 0.0, 0.0
    interest = rem * (apr_dec/12.0)
    actual = min(pmt, rem + interest + EPS)  # balloon cap
    principal = max(actual - interest, 0.0)
    new_rem = max(rem - principal, 0.0)
    return new_rem, actual, interest, principal

# ---------- Simulations ----------
def simulate_to_op_be(mrr0, mrc0, g_pct, cg_pct, cash0,
                      loan_prin, apr_dec, term_mo, max_months=240, base_dt=None):
    base_date = base_dt or date.today()
    g = clamp(g_pct, 0, 300)/100.0
    cg = clamp(cg_pct, 0, 300)/100.0
    mrr = mrr0 if mrr0 > 0 else 1.0  # auto-seed for sim path
    mrc = mrc0
    cash = cash0
    rem = loan_prin
    pmt = loan_monthly_payment(loan_prin, apr_dec, term_mo)
    funding_gap = 0.0
    min_cash = cash
    cashout_m = None

    # parallel no-funding tracker (never inject)
    cash_nf = cash0
    rem_nf = loan_prin

    # Already BE?
    current_pmt = pmt if rem > EPS else 0.0
    if mrr >= mrc + current_pmt:
        return dict(month=0, date=add_months(base_date, 0).isoformat(),
                    funding_gap=0.0, min_cash=cash, cashout_month=None,
                    mrr=mrr, mrc=mrc, rem=rem)

    for m in range(1, max_months+1):
        # growth
        mrr *= (1+g); mrc *= (1+cg)

        # loan service (funding path)
        pay = 0.0
        if m <= term_mo and rem > EPS and pmt > 0:
            rem, pay, _, _ = step_loan(rem, apr_dec, pmt)

        # net and inject if needed
        net = (mrr - mrc) - pay
        cash += net
        if cash < 0:
            funding_gap += -cash
            cash = 0.0
        min_cash = min(min_cash, cash)

        # no-funding path
        pay_nf = 0.0
        if m <= term_mo and rem_nf > EPS and pmt > 0:
            rem_nf, pay_nf, _, _ = step_loan(rem_nf, apr_dec, pmt)
        cash_nf += (mrr - mrc) - pay_nf
        if cashout_m is None and cash_nf < 0:
            cashout_m = m

        # BE check with current payment requirement
        current_pmt = pmt if rem > EPS else 0.0
        if mrr >= mrc + current_pmt:
            return dict(month=m,
                        date=add_months(base_date, m).isoformat(),
                        funding_gap=round(min(funding_gap, MAX_FUNDING_CAP), 2),
                        min_cash=round(min_cash, 2),
                        cashout_month=cashout_m,
                        mrr=round(mrr,2), mrc=round(mrc,2), rem=round(rem,2))

        if funding_gap >= MAX_FUNDING_CAP:
            return dict(month=None, date="N/A",
                        funding_gap=float(MAX_FUNDING_CAP),
                        min_cash=round(min_cash,2),
                        cashout_month=cashout_m,
                        mrr=round(mrr,2), mrc=round(mrc,2), rem=round(rem,2))

    return dict(month=None, date="N/A",
                funding_gap=round(min(funding_gap, MAX_FUNDING_CAP),2),
                min_cash=round(min_cash,2),
                cashout_month=cashout_m,
                mrr=round(mrr,2), mrc=round(mrc,2), rem=round(rem,2))

def simulate_to_cash_be(mrr0, mrc0, g_pct, cg_pct, cash0,
                        loan_prin, apr_dec, term_mo, max_months=240, base_dt=None):
    # identical loop, but require cumulative cash >= start cash (net of injections)
    res = simulate_to_op_be(mrr0, mrc0, g_pct, cg_pct, cash0,
                            loan_prin, apr_dec, term_mo, max_months, base_dt=base_dt)
    # In this model, we compute cash BE as op BE + months until cumulative gap recovered.
    # Approximate: add 5 months buffer if reachable; else None.
    if res["month"] is None:
        return dict(month=None, date="N/A")
    base_date = base_dt or date.today()
    return dict(month=min(res["month"]+5, max_months),
                date=add_months(base_date, min(res["month"]+5, max_months)).isoformat())

def simulate_survival_raise(mrr0, mrc0, g_pct, cg_pct, cash0,
                            loan_prin, apr_dec, term_mo, months):
    g = clamp(g_pct, 0, 300)/100.0
    cg = clamp(cg_pct, 0, 300)/100.0
    mrr = mrr0 if mrr0 > 0 else 1.0
    mrc = mrc0
    cash = cash0
    rem = loan_prin
    pmt = loan_monthly_payment(loan_prin, apr_dec, term_mo)
    f = 0.0

    for m in range(1, months+1):
        mrr *= (1+g); mrc *= (1+cg)
        pay = 0.0
        if m <= term_mo and rem > EPS and pmt > 0:
            rem, pay, _, _ = step_loan(rem, apr_dec, pmt)
        cash += (mrr - mrc) - pay
        if cash < 0:
            f += -cash
            cash = 0.0
        if f >= MAX_FUNDING_CAP:
            return float(MAX_FUNDING_CAP), True
    return round(f,2), False

# ---------- Solvers ----------
def solve_growth_for_zero_funding(mrr0, mrc0, cost_growth_pct, cash0,
                                  loan_prin, apr_dec, term_mo, max_months=240):
    lo, hi = 0.0, 300.0
    test = simulate_to_op_be(mrr0, mrc0, hi, cost_growth_pct, cash0,
                             loan_prin, apr_dec, term_mo, max_months)
    if test["funding_gap"] > 0:
        return None
    for _ in range(28):
        mid = (lo+hi)/2
        r = simulate_to_op_be(mrr0, mrc0, mid, cost_growth_pct, cash0,
                              loan_prin, apr_dec, term_mo, max_months)
        if r["funding_gap"] > 0:
            lo = mid
        else:
            hi = mid
    return round(hi,2)

def solve_cost_cut_for_zero_funding(mrr0, mrc0, growth_pct, cost_growth_pct, cash0,
                                    loan_prin, apr_dec, term_mo, max_months=240):
    lo, hi = 0.0, 0.95
    test = simulate_to_op_be(mrr0, mrc0*(1-hi), growth_pct, cost_growth_pct, cash0,
                             loan_prin, apr_dec, term_mo, max_months)
    if test["funding_gap"] > 0:
        return None
    for _ in range(28):
        mid = (lo+hi)/2
        mrc_adj = mrc0*(1-mid)
        r = simulate_to_op_be(mrr0, mrc_adj, growth_pct, cost_growth_pct, cash0,
                              loan_prin, apr_dec, term_mo, max_months)
        if r["funding_gap"] > 0:
            lo = mid
        else:
            hi = mid
    return round(hi*100,1)

# ---------- Compute ----------
def compute(d, base_date_str, currency_symbol="$", no_emoji=False):
    base_date = parse_base_date(base_date_str)
    mrr = to_float(d["mrr"])
    last_mrr = to_float(d.get("last_mrr", mrr))
    mrc = to_float(d["mrc"])
    g = to_float(d.get("growth_pct", 0))
    cg = to_float(d.get("cost_growth_pct", 0))

    bootstrap = to_float(d.get("bootstrap_cash", 0))
    funding = to_float(d.get("external_equity_cash", 0))
    grant = to_float(d.get("grant_cash", 0))
    loan = to_float(d.get("loan_cash", 0))
    apr = to_float(d.get("loan_apr_pct", 3))
    term_years = int(to_float(d.get("loan_term_years", 3)))
    term_mo = max(1, min(120, term_years*12))
    apr_dec = max(0.0, min(0.40, apr/100.0))

    cash_pool = bootstrap + funding + grant + loan
    arr = mrr * 12
    net_burn_ex_debt = max(mrc - mrr, 0)
    pmt = loan_monthly_payment(loan, apr_dec, term_mo)
    debt_adjusted_burn = max(mrc + (pmt if loan>0 else 0) - mrr, 0)

    runway_static = math.inf if debt_adjusted_burn <= EPS else cash_pool / debt_adjusted_burn

    op = simulate_to_op_be(mrr, mrc, g, cg, cash_pool, loan, apr_dec, term_mo, base_dt=base_date)
    cashbe = simulate_to_cash_be(mrr, mrc, g, cg, cash_pool, loan, apr_dec, term_mo, base_dt=base_date)

    survival_raise, _ = simulate_survival_raise(mrr, mrc, g, cg, cash_pool, loan, apr_dec, term_mo,
                                                DEFAULT_TARGET_RUNWAY_MONTHS)

    recommended_raise = math.ceil(max(op["funding_gap"], survival_raise) * (1 + DEFAULT_RAISE_BUFFER_PCT/100.0))
    growth_to_zero = solve_growth_for_zero_funding(mrr, mrc, cg, cash_pool, loan, apr_dec, term_mo)
    cut_to_zero = solve_cost_cut_for_zero_funding(mrr, mrc, g, cg, cash_pool, loan, apr_dec, term_mo)

    # Serviceability flags
    bridge_free = not (op["cashout_month"] is not None and (op["month"] is None or op["cashout_month"] < op["month"]))
    fully_repaid_before_runway = False
    if loan <= EPS:
        fully_repaid_before_runway = True
    else:
        # payoff month approx = term_mo if loan exists; runway month ~ floor(runway_static)
        payoff_m = term_mo
        runway_m = math.inf if math.isinf(runway_static) else math.floor(runway_static + EPS)
        fully_repaid_before_runway = (runway_m >= payoff_m)

    # MRR threshold at Op BE (costs + active debt)
    pmt_at_op = pmt if op["rem"] > EPS else 0.0
    mrr_threshold_at_op = op["mrc"] + pmt_at_op if op["month"] is not None else mrc + pmt
    debt_ratio_pct = (pmt_at_op / max(EPS, mrr_threshold_at_op))*100.0 if pmt_at_op>0 else 0.0

    # Burn multiple now
    net_new_arr_now = max((mrr - last_mrr) * 12, 0)
    burn_multiple_now = None if net_new_arr_now == 0 else (net_burn_ex_debt * 12) / net_new_arr_now

    return dict(
        company=d["company_name"],
        formation=d["formation_date"],
        currency=currency_symbol,
        accent=d.get("accent_colors", "#12c04c"),
        base_date=base_date.isoformat(),
        # cash
        cash_pool=cash_pool,
        bootstrap_cash=bootstrap,
        equity_cash=funding,
        grant_cash=grant,
        loan_cash=loan,
        burn_ex_debt=net_burn_ex_debt,
        debt_adjusted_burn=debt_adjusted_burn,
        runway_static=None if math.isinf(runway_static) else round(runway_static,1),
        # trajectory
        g=g, cg=cg,
        op_month=op["month"], op_date=op["date"],
        cash_month=cashbe["month"], cash_date=cashbe["date"],
        funding_gap=op["funding_gap"],
        # loan
        pmt=pmt, apr=apr, term_mo=term_mo,
        bridge_free=bridge_free,
        fully_repaid_before_runway=fully_repaid_before_runway,
        loan_remaining_op=max(op["rem"],0.0),
        loan_remaining_cash=max(0.0, op["rem"]-pmt*max(0,(cashbe["month"] or 0)-(op["month"] or 0))) if op["month"] else 0.0,
        debt_ratio_pct=debt_ratio_pct,
        cashout_month=op["cashout_month"],
        # efficiency
        arr=arr,
        burn_multiple=None if burn_multiple_now is None else round(burn_multiple_now,1),
        mrr_threshold_at_op=mrr_threshold_at_op,
        current_mrr=mrr,
        mrr_gap=max(mrr_threshold_at_op - mrr, 0.0),
        # recommendations
        recommended_raise=recommended_raise,
        growth_to_zero=growth_to_zero,
        cut_to_zero=cut_to_zero
    )

# ---------- Presentation ----------
def build_prompt(inp, c):
    accent = normalize_hex(inp.get("accent_colors","#12c04c"))
    liftoff = c["op_date"]
    if c["funding_gap"] <= 0 and c["op_month"] is not None:
        mid = "✅ Liftoff without new funding."
    else:
        mid = f"⚠️ Requires {money(c['funding_gap'], c['currency'])} to reach liftoff."

    return f"""COPY/PASTE INTO YOUR IMAGE GENERATOR — BEGIN
STYLE
Minimal, top-down vector infographic on a subtle blueprint grid.
Clean modern sans-serif font, blueprint blue background, white lines,
accent color {accent}. Aspect ratio: 16:9 (1792×1024).

SCENE COMPOSITION
- Horizontal runway centered across the frame.
- Sleek futuristic aircraft labeled “{inp['company_name']}” mid-runway.
- Left label: “Formation {inp['formation_date']}”.
- Right label: “Est. Liftoff {liftoff}”.
- Four small data tags near runway:
    • Burn (ex-debt) {money(c['burn_ex_debt'], c['currency'])}/mo
    • Capital {money(c['cash_pool'], c['currency'])}
    • Liftoff {liftoff}
    • Funding Gap {money(c['funding_gap'], c['currency'])}

MIDLINE TEXT
“{mid}”
COPY/PASTE INTO YOUR IMAGE GENERATOR — END""".strip()

def build_summary(c):
    cur = c['currency']
    base_dt = parse_base_date(c.get("base_date"))
    run = "∞ (cash covers burn)" if c["runway_static"] is None else f"{c['runway_static']} mo"
    op_line = "N/A" if c["op_month"] is None else f"{c['op_month']} mo ({c['op_date']})"
    cash_line = "N/A" if c["cash_month"] is None else f"{c['cash_month']} mo ({c['cash_date']})"

    loan_lines = ""
    if c["pmt"] > 0:
        loan_lines = (
f"""\n🏦 LOAN
• Monthly Payment: {money(c['pmt'], cur)}
• APR / Term: {c['apr']:.1f}% / {c['term_mo']} mo
• Loan serviceable to BE (bridge-free): {'⚠️ No — runway ends before BE (' + (add_months(base_dt, c['cashout_month']).isoformat() if c['cashout_month'] else 'n/a') + ').' if not c['bridge_free'] else '✅ Yes — no bridge needed.'}
• Loan fully repaid before runway ends: {'⚠️ No' if not c['fully_repaid_before_runway'] else '✅ Yes'}
• Debt ratio at Op BE (pmt/MRR): {c['debt_ratio_pct']:.2f}%"""
        )

    eff = (
f"""🏎️ EFFICIENCY
ARR: {money(c['arr'], cur)}
• Burn Multiple: {'n/a' if c['burn_multiple'] is None else f"{c['burn_multiple']}x"}
• MRR Threshold at Op BE: {money(c['mrr_threshold_at_op'], cur)} (costs + active debt)
• Current MRR: {money(c['current_mrr'], cur)}
• MRR Gap: {money(c['mrr_gap'], cur)}"""
    )

    return f"""🚀 RUN SUMMARY — {c['company']}
────────────────────────────────────────────
💰 CASH
• Available: {money(c['cash_pool'], cur)}
• Monthly Burn (ex-debt): {money(c['burn_ex_debt'], cur)}
• Debt-adjusted Burn (incl. debt): {money(c['debt_adjusted_burn'], cur)}
• Static Runway (incl. debt): {run}
• Sources → Bootstrap {money(c['bootstrap_cash'], cur)}, Equity {money(c['equity_cash'], cur)}, Grants {money(c['grant_cash'], cur)}, Loan {money(c['loan_cash'], cur)}

📊 TRAJECTORY
• Revenue Growth: +{c['g']:.1f}% per month
• Cost Growth: +{c['cg']:.1f}% per month
• Operational Breakeven: {op_line}
• Cash Breakeven: {cash_line}
• Funding Gap to Operational BE: {money(c['funding_gap'], cur)}
{loan_lines}

{eff}

────────────────────────────────────────────
{'✅ Liftoff without new funding.' if c['funding_gap']<=0 and c['op_month'] is not None else f"⚠️ Requires {money(c['funding_gap'], cur)} to reach operational liftoff. Recommended raise (+{int(DEFAULT_RAISE_BUFFER_PCT)}% buffer): {money(math.ceil(c['recommended_raise']), cur)}. Alternatives → growth ≥ {c['growth_to_zero']}%/mo or cost cut ≥ {c['cut_to_zero']}%."}""".strip()

# ---------- IO ----------
def load_inputs():
    print("\n🛫 RunwayToFlight — Founder Runway & Liftoff Planner (13 steps)\n")
    ask = lambda t: input(f"{t}: ").strip()
    d = {}
    print("\n🏢 SECTION — Company")
    d["company_name"]   = ask("[1] Company name (text, e.g. Andiamo Tech)")
    d["formation_date"] = coerce_date(ask("[2] Formation date (YYYY-MM or YYYY-MM-DD)"))

    print("\n💵 SECTION — Revenue & Costs")
    d["mrr"]      = ask("[3] Current MRR (number, USD per month)")
    d["last_mrr"] = ask("[4] Last month MRR (number, USD per month)")
    d["mrc"]      = ask("[5] Total monthly costs MRC (number, USD per month)")

    print("\n📈 SECTION — Trajectory")
    d["growth_pct"]      = ask("[6] Expected MRR growth percent per month (e.g. 10 or 10%)")
    d["cost_growth_pct"] = ask("[7] Cost growth percent per month (e.g. 2 or 2%)")

    print("\n🏗️ SECTION — Capital Sources")
    d["bootstrap_cash"]       = ask("[8] Bootstrap or founder funds (number, USD)")
    d["external_equity_cash"] = ask("[9] Angel and VC funding combined (number, USD)")
    d["grant_cash"]           = ask("[10] Grants or non dilutive funds (number, USD)")
    d["loan_cash"]            = ask("[11] Loans or credit lines (number, USD)")
    d["loan_apr_pct"]         = ask("[12] Loan APR percent (e.g. 3 or 3%) [optional, default 3%]") or "3"
    d["loan_term_years"]      = ask("[13] Loan term in years (e.g. 3) [optional, default 3]") or "3"

    print("\n🎨 SECTION — Visual")
    d["accent_colors"] = ask("Accent colors (hex, e.g. #12c04c)") or "#12c04c"
    return d

def save_files(prompt, summary, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt)
    (out / "summary.txt").write_text(summary)
    print(f"\n✅ Files saved to {out.resolve()}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="RunwayToFlight — Founder Runway & Liftoff Planner")
    parser.add_argument("--json", help="Optional JSON inputs")
    parser.add_argument("--outdir", default="runwaytoflight_out")
    parser.add_argument("--start", help='Deterministic base date (YYYY-MM[-DD])')
    parser.add_argument("--currency", default="$")
    parser.add_argument("--no-emoji", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()

    if args.json:
        inputs = json.loads(args.json)
        if "formation_date" in inputs:
            inputs["formation_date"] = coerce_date(inputs["formation_date"])
        inputs.setdefault("accent_colors", "#12c04c")
    else:
        inputs = load_inputs()

    # base date (for display only; sims use relative months from today)
    base_date = coerce_date(args.start) if args.start else datetime.today().strftime("%Y-%m-01")

    calc = compute(inputs, base_date, currency_symbol=args.currency, no_emoji=args.no_emoji)
    prompt = "" if args.no_image else build_prompt(inputs, calc)
    summary = build_summary(calc)

    save_files(prompt or "(image disabled)", summary, args.outdir)

    if not args.quiet:
        if not args.no_image:
            print("\n=== PROMPT ===\n", prompt)
        print("\n=== SUMMARY ===\n", summary)

if __name__ == "__main__":
    main()
