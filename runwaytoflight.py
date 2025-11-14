from __future__ import annotations
from pathlib import Path
import argparse, json, math
from datetime import date, datetime

# RunwayToFlight v3.5 — Founder Runway & Liftoff Planner
# Debt-aware runway, breakeven, and funding gap simulator.
# CLI + library-friendly (compute, build_prompt, build_summary, coerce_date).

DEFAULT_TARGET_RUNWAY_MONTHS = 24
DEFAULT_RAISE_BUFFER_PCT = 20.0
MAX_FUNDING_CAP = 10_000_000.0
EPS = 1e-6

# ----------------- Utils -----------------

def clamp(v: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, x))

def to_float(v) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return 0.0

def money(v: float) -> str:
    x = float(v)
    return f"{x:,.2f}"

def coerce_date(s: str | None) -> str:
    """
    Accepts YYYY-MM-DD or YYYY-MM and normalizes to YYYY-MM-01.
    Returns the original string if parsing fails.
    """
    if not s:
        return ""
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return s

def parse_base_date(base_date_str: str | None) -> date:
    if base_date_str:
        cd = coerce_date(base_date_str)
        try:
            return datetime.strptime(cd, "%Y-%m-%d").date()
        except Exception:
            pass
    # Fallback: first of this month
    today = date.today()
    return date(today.year, today.month, 1)

def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
             31,30,31,30,31,31,30,31,30,31][m-1]
    return date(y, m, min(d.day, mdays))

def normalize_hex(c: str | None) -> str:
    c = (c or "").strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return f"#{c}" if c else "#12c04c"

def em(flag: bool, no_emoji: bool) -> str:
    if no_emoji:
        return "YES" if flag else "NO"
    return "✅" if flag else "⚠️"

# ----------------- Loan helpers -----------------

def loan_monthly_payment(principal: float, apr_decimal: float, term_months: int) -> float:
    if principal <= 0 or term_months <= 0 or apr_decimal <= 0:
        return 0.0
    r = apr_decimal / 12.0
    return principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)

def step_loan(rem: float, apr_decimal: float, pmt: float) -> tuple[float, float, float, float]:
    """
    Single month of amortization.
    Caps final payment at remaining principal + interest.
    Returns: (new_remaining, actual_payment, interest, principal_paid)
    """
    if rem <= EPS or pmt <= 0 or apr_decimal <= 0:
        return rem, 0.0, 0.0, 0.0
    interest = rem * (apr_decimal / 12.0)
    due = rem + interest
    # EPS tolerance avoids float edge cases on final payment
    pay = min(pmt, due + EPS)
    principal_paid = max(pay - interest, 0.0)
    new_rem = max(rem - principal_paid, 0.0)
    if new_rem < EPS:
        new_rem = 0.0
    return new_rem, pay, interest, principal_paid

# ----------------- Core simulations -----------------

def simulate_to_op_be(
    mrr0: float,
    mrc0: float,
    g_pct: float,
    cg_pct: float,
    total_cash: float,
    loan_principal: float,
    apr_decimal: float,
    term_months: int,
    base_date: date,
) -> dict:
    g = g_pct / 100.0
    cg = cg_pct / 100.0
    auto_seeded = mrr0 <= 0
    mrr = 1.0 if auto_seeded else mrr0
    mrc = mrc0

    rem = loan_principal
    pmt = loan_monthly_payment(loan_principal, apr_decimal, term_months)
    cash = total_cash
    funding_gap = 0.0
    op_month = None
    op_date = None
    cashout_month = None
    cashout_date = None
    payoff_month = None
    mrr_at_op = None
    mrc_at_op = None
    pmt_at_op = 0.0
    unreachable = False

    for m in range(1, 241):
        # Apply growth
        mrr *= (1 + g)
        mrc *= (1 + cg)

        # Loan service
        current_pmt = 0.0
        if m <= term_months and rem > EPS and pmt > 0:
            rem, pay, _, _ = step_loan(rem, apr_decimal, pmt)
            current_pmt = pay
            cash -= pay
            if rem <= EPS and payoff_month is None:
                payoff_month = m

        # Operating cashflow
        net = mrr - mrc
        cash += net

        # If cash dips below zero, inject just enough to reset to 0
        if cash < 0:
            if cashout_month is None:
                cashout_month = m
                cashout_date = add_months(base_date, m)
            funding_gap += -cash
            cash = 0.0
            if funding_gap >= MAX_FUNDING_CAP:
                unreachable = True
                op_month = None
                op_date = None
                break

        # Check operational breakeven (MRR covers costs + active debt)
        if mrr >= mrc + current_pmt and op_month is None:
            op_month = m
            op_date = add_months(base_date, m)
            mrr_at_op = mrr
            mrc_at_op = mrc
            pmt_at_op = current_pmt
            break

    if op_month is None and not unreachable:
        unreachable = True

    return dict(
        op_month=op_month,
        op_date=op_date.isoformat() if op_date else None,
        funding_gap=min(funding_gap, MAX_FUNDING_CAP),
        unreachable=unreachable,
        cashout_month=cashout_month,
        cashout_date=cashout_date.isoformat() if cashout_date else None,
        payoff_month=payoff_month,
        mrr_at_op=mrr_at_op,
        mrc_at_op=mrc_at_op,
        pmt_at_op=pmt_at_op,
        auto_seeded=auto_seeded,
    )

def simulate_to_cash_be(
    mrr0: float,
    mrc0: float,
    g_pct: float,
    cg_pct: float,
    total_cash: float,
    loan_principal: float,
    apr_decimal: float,
    term_months: int,
    funding_gap: float,
    base_date: date,
) -> dict:
    g = g_pct / 100.0
    cg = cg_pct / 100.0
    mrr = 1.0 if mrr0 <= 0 else mrr0
    mrc = mrc0

    rem = loan_principal
    pmt = loan_monthly_payment(loan_principal, apr_decimal, term_months)

    # Start with original cash + the gap raised upfront
    start_cash = total_cash + max(funding_gap, 0.0)
    cash = start_cash
    cash_be_month = None
    cash_be_date = None
    payoff_month = None

    for m in range(1, 241):
        mrr *= (1 + g)
        mrc *= (1 + cg)

        # Loan service
        if m <= term_months and rem > EPS and pmt > 0:
            rem, pay, _, _ = step_loan(rem, apr_decimal, pmt)
            cash -= pay
            if rem <= EPS and payoff_month is None:
                payoff_month = m

        net = mrr - mrc
        cash += net

        if cash_be_month is None and cash >= start_cash:
            cash_be_month = m
            cash_be_date = add_months(base_date, m)
            break

    return dict(
        cash_month=cash_be_month,
        cash_date=cash_be_date.isoformat() if cash_be_date else None,
        payoff_month=payoff_month,
    )

# ----------------- Solvers -----------------

def solve_growth_for_zero_funding(
    mrr0: float,
    mrc0: float,
    cost_growth_pct: float,
    total_cash: float,
    loan_principal: float,
    apr_decimal: float,
    term_months: int,
    base_date: date,
) -> float | None:
    hi = 300.0
    # If even at hi growth we still need funding and/or unreachable, return None
    test = simulate_to_op_be(mrr0, mrc0, hi, cost_growth_pct, total_cash, loan_principal, apr_decimal, term_months, base_date)
    if test["unreachable"] or test["funding_gap"] > 0:
        return None

    lo = 0.0
    for _ in range(28):
        mid = (lo + hi) / 2.0
        res = simulate_to_op_be(mrr0, mrc0, mid, cost_growth_pct, total_cash, loan_principal, apr_decimal, term_months, base_date)
        if res["unreachable"] or res["funding_gap"] > 0:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)

def solve_cost_cut_for_zero_funding(
    mrr0: float,
    mrc0: float,
    growth_pct: float,
    cost_growth_pct: float,
    total_cash: float,
    loan_principal: float,
    apr_decimal: float,
    term_months: int,
    base_date: date,
) -> float | None:
    hi = 0.95  # Up to 95% immediate cost cut
    # If even at aggressive cut we still need funding / unreachable, return None
    test = simulate_to_op_be(mrr0, mrc0 * (1 - hi), growth_pct, cost_growth_pct, total_cash, loan_principal, apr_decimal, term_months, base_date)
    if test["unreachable"] or test["funding_gap"] > 0:
        return None

    lo = 0.0
    for _ in range(28):
        mid = (lo + hi) / 2.0
        mrc_adj = mrc0 * (1 - mid)
        res = simulate_to_op_be(mrr0, mrc_adj, growth_pct, cost_growth_pct, total_cash, loan_principal, apr_decimal, term_months, base_date)
        if res["unreachable"] or res["funding_gap"] > 0:
            lo = mid
        else:
            hi = mid
    return round(hi * 100.0, 1)

# ----------------- Compute -----------------

def compute(d: dict, base_date_str: str | None = None, no_emoji: bool = False) -> dict:
    company_name = d.get("company_name", "").strip() or "Company"
    formation_date = coerce_date(d.get("formation_date", ""))

    mrr = to_float(d.get("mrr", 0))
    last_mrr = to_float(d.get("last_mrr", mrr))
    mrc = to_float(d.get("mrc", 0))

    g = clamp(d.get("growth_pct", 0), 0, 300)
    cg = clamp(d.get("cost_growth_pct", 0), 0, 300)

    bootstrap = to_float(d.get("bootstrap_cash", 0))
    equity = to_float(d.get("external_equity_cash", 0))
    grant = to_float(d.get("grant_cash", 0))
    loan = to_float(d.get("loan_cash", 0))

    apr_pct = clamp(d.get("loan_apr_pct", 3.0), 0, 40)
    term_years = clamp(d.get("loan_term_years", 3.0), 0, 120/12)
    # Use round to avoid truncation (e.g., 3.5 years -> 42 months)
    term_months = int(round(term_years * 12))

    total_cash = bootstrap + equity + grant + loan
    base_date = parse_base_date(base_date_str or formation_date or None)

    apr_decimal = apr_pct / 100.0 if loan > 0 and apr_pct > 0 and term_months > 0 else 0.0
    pmt = loan_monthly_payment(loan, apr_decimal, term_months) if apr_decimal > 0 else 0.0

    # Core burns
    burn_ex_debt = max(mrc - mrr, 0.0)
    burn_incl_debt = burn_ex_debt + pmt
    static_runway = None if burn_incl_debt <= 0 else total_cash / burn_incl_debt

    # Primary op BE simulation
    op_res = simulate_to_op_be(mrr, mrc, g, cg, total_cash, loan, apr_decimal, term_months, base_date)
    op_month = op_res["op_month"]
    op_date = op_res["op_date"]
    funding_gap = op_res["funding_gap"]
    unreachable = op_res["unreachable"]
    cashout_month = op_res["cashout_month"]
    cashout_date = op_res["cashout_date"]
    payoff_month_op = op_res["payoff_month"]
    mrr_at_op = op_res["mrr_at_op"]
    mrc_at_op = op_res["mrc_at_op"]
    pmt_at_op = op_res["pmt_at_op"]
    auto_seeded = op_res["auto_seeded"]

    # Cash BE simulation (only if we got a finite funding gap and op BE)
    cash_res = {"cash_month": None, "cash_date": None, "payoff_month": None}
    if not unreachable and funding_gap < MAX_FUNDING_CAP and op_month is not None:
        cash_res = simulate_to_cash_be(mrr, mrc, g, cg, total_cash, loan, apr_decimal, term_months, funding_gap, base_date)

    cash_month = cash_res["cash_month"]
    cash_date = cash_res["cash_date"]
    payoff_month_cash = cash_res["payoff_month"]

    # Is loan serviceable without extra bridge to BE?
    be_reachable_bridge_free = (op_month is not None) and (
        cashout_month is None or cashout_month >= op_month
    )
    loan_fully_before_runway = False
    payoff_month = None
    if loan > 0 and apr_decimal > 0 and term_months > 0:
        # earliest payoff across sims
        candidates = [m for m in (payoff_month_op, payoff_month_cash) if m is not None]
        payoff_month = min(candidates) if candidates else None
        if payoff_month is not None and cashout_month is not None:
            loan_fully_before_runway = payoff_month <= cashout_month

    # Funding cap handling
    if funding_gap >= MAX_FUNDING_CAP:
        unreachable = True
        funding_gap = MAX_FUNDING_CAP

    # ARR and burn multiple
    arr = mrr * 12.0
    net_new_arr = max((mrr - last_mrr) * 12.0, 0.0)
    burn_multiple = None
    if net_new_arr > 0 and burn_ex_debt > 0:
        burn_multiple = (burn_ex_debt * 12.0) / net_new_arr

    # Survival raise for target runway: rough heuristic vs gap
    survival_raise = 0.0
    if burn_incl_debt > 0:
        survival_raise = burn_incl_debt * DEFAULT_TARGET_RUNWAY_MONTHS
    recommended_raise = math.ceil(max(funding_gap, survival_raise) * (1 + DEFAULT_RAISE_BUFFER_PCT / 100.0))

    # Alternative levers
    growth_to_zero = solve_growth_for_zero_funding(
        mrr, mrc, cg, total_cash, loan, apr_decimal, term_months, base_date
    )
    cut_to_zero = solve_cost_cut_for_zero_funding(
        mrr, mrc, g, cg, total_cash, loan, apr_decimal, term_months, base_date
    )

    # MRR threshold at op BE
    mrr_threshold = None
    debt_ratio = None
    if mrr_at_op is not None and mrc_at_op is not None:
        mrr_threshold = mrc_at_op + (pmt_at_op or 0.0)
        if mrr_at_op > 0:
            debt_ratio = (pmt_at_op or 0.0) / mrr_at_op

    return dict(
        company_name=company_name,
        formation_date=formation_date,
        base_date=base_date.isoformat(),
        mrr=mrr,
        last_mrr=last_mrr,
        mrc=mrc,
        growth_pct=g,
        cost_growth_pct=cg,
        bootstrap=bootstrap,
        equity=equity,
        grant=grant,
        loan=loan,
        apr_pct=apr_pct,
        term_months=term_months,
        total_cash=total_cash,
        burn_ex_debt=burn_ex_debt,
        burn_incl_debt=burn_incl_debt,
        static_runway=None if static_runway is None else round(static_runway, 1),
        op_month=op_month,
        op_date=op_date,
        cash_month=cash_month,
        cash_date=cash_date,
        funding_gap=funding_gap,
        unreachable=unreachable,
        cashout_month=cashout_month,
        cashout_date=cashout_date,
        payoff_month=payoff_month,
        be_reachable_bridge_free=be_reachable_bridge_free,
        loan_fully_before_runway=loan_fully_before_runway,
        arr=arr,
        burn_multiple=None if burn_multiple is None else round(burn_multiple, 1),
        survival_raise=survival_raise,
        recommended_raise=recommended_raise,
        growth_to_zero=growth_to_zero,
        cut_to_zero=cut_to_zero,
        mrr_threshold=mrr_threshold,
        mrr_at_op=mrr_at_op,
        debt_ratio=debt_ratio,
        auto_seeded=auto_seeded,
    )

# ----------------- Output builders -----------------

def build_prompt(inp: dict, c: dict, currency_symbol: str = "$") -> str:
    company = c["company_name"]
    accent = normalize_hex(inp.get("accent_colors", "#12c04c"))
    liftoff = c["op_date"] or "N/A"

    burn_tag = f"{currency_symbol}{money(c['burn_ex_debt'])}/mo"
    cap_tag = f"{currency_symbol}{money(c['total_cash'])}"
    gap_tag = f"{currency_symbol}{money(c['funding_gap'])}"

    if not c["unreachable"] and c["op_month"] is not None and c["funding_gap"] <= 0:
        mid_text = f"✅ Liftoff in {c['op_month']} mo ({liftoff}) without new funding."
    elif c["funding_gap"] >= MAX_FUNDING_CAP or c["unreachable"]:
        mid_text = (
            "❌ Cannot reach operational liftoff with these inputs. "
            f"Model estimates more than {currency_symbol}{money(MAX_FUNDING_CAP)} required."
        )
    else:
        mid_text = f"⚠️ Requires {currency_symbol}{money(c['funding_gap'])} to reach liftoff."

    formation = inp.get("formation_date", "")
    return f"""COPY/PASTE INTO YOUR IMAGE GENERATOR — BEGIN
STYLE
Minimal, top-down vector infographic on a subtle blueprint grid.
Clean modern sans-serif font, blueprint blue background, white lines,
accent color {accent}. Aspect ratio: 16:9 (1792×1024).

SCENE COMPOSITION
- Horizontal runway centered across the frame.
- Sleek futuristic aircraft labeled “{company}” mid-runway.
- Left label: “Formation {formation}”.
- Right label: “Est. Liftoff {liftoff}”.
- Four small data tags near runway:
    • Burn (ex-debt) {burn_tag}
    • Capital {cap_tag}
    • Liftoff {liftoff}
    • Funding Gap {gap_tag}

MIDLINE TEXT
“{mid_text}”
COPY/PASTE INTO YOUR IMAGE GENERATOR — END""".strip()

def build_bottom_line(c: dict, currency_symbol: str = "$") -> str:
    gap = c["funding_gap"]
    g_opt = c.get("growth_to_zero")
    c_opt = c.get("cut_to_zero")

    if g_opt is None:
        g_str = "N/A"
    else:
        g_str = f"{g_opt:.2f}%"

    if c_opt is None:
        c_str = "N/A"
    else:
        c_str = f"{c_opt:.1f}%"

    if gap >= MAX_FUNDING_CAP or c.get("unreachable"):
        return (
            "❌ Cannot reach operational liftoff with these inputs. "
            f"Model estimates more than {currency_symbol}{money(MAX_FUNDING_CAP)} required, "
            "which is beyond the scope of this tool."
        )

    rec = c["recommended_raise"]
    return (
        f"⚠️ Requires {currency_symbol}{money(gap)} to reach operational liftoff. "
        f"Recommended raise (+{int(DEFAULT_RAISE_BUFFER_PCT)}% buffer): {currency_symbol}{money(rec)}. "
        f"Alternatives → growth ≥ {g_str}/mo or cost cut ≥ {c_str}."
    )

def build_summary(c: dict, currency_symbol: str = "$", no_emoji: bool = False) -> str:
    company = c["company_name"]
    cash = c["total_cash"]
    burn = c["burn_ex_debt"]
    burn_incl = c["burn_incl_debt"]
    runway = c["static_runway"]
    arr = c["arr"]
    bm = c["burn_multiple"]
    loan = c["loan"]
    pmt = loan_monthly_payment(loan, c["apr_pct"]/100.0, c["term_months"]) if loan > 0 and c["apr_pct"] > 0 and c["term_months"] > 0 else 0.0

    base_runway = "∞ (cash covers burn)" if runway is None else f"{runway:.1f} mo"
    bm_str = "n/a" if bm is None else f"{bm:.1f}x"
    arr_str = f"{currency_symbol}{money(arr)}"

    op_str = "N/A"
    if c["op_month"] is not None and c["op_date"]:
        op_str = f"{c['op_month']} mo ({c['op_date']})"

    cash_str = "N/A"
    if c["cash_month"] is not None and c["cash_date"]:
        cash_str = f"{c['cash_month']} mo ({c['cash_date']})"

    # Loan section
    loan_lines = ""
    if loan > 0 and pmt > 0:
        serviceable = c["be_reachable_bridge_free"]
        fully_before_runway = c["loan_fully_before_runway"]
        dr = c["debt_ratio"]
        dr_str = "n/a" if dr is None else f"{dr * 100:.2f}%"

        if c["cashout_month"] is not None and c["cashout_date"]:
            service_line = f"{em(False, no_emoji)} No — runway ends before BE ({c['cashout_date']})."
        else:
            service_line = f"{em(serviceable, no_emoji)} {'Yes' if serviceable else 'No'}."

        fully_line = f"{em(fully_before_runway, no_emoji)} {'Yes' if fully_before_runway else 'No'}"

        loan_lines = f"""
🏦 LOAN
• Monthly Payment: {currency_symbol}{money(pmt)}
• APR / Term: {c['apr_pct']:.1f}% / {c['term_months']} mo
• Loan serviceable to BE (bridge-free): {service_line}
• Loan fully repaid before runway ends: {fully_line}
• Debt ratio at Op BE (pmt/MRR): {dr_str}
"""

    mrr_threshold = c.get("mrr_threshold")
    mrr_threshold_str = "n/a" if mrr_threshold is None else f"{currency_symbol}{money(mrr_threshold)}"
    mrr_current_str = f"{currency_symbol}{money(c['mrr'])}"
    mrr_gap_str = "n/a"
    if mrr_threshold is not None:
        mrr_gap_str = f"{currency_symbol}{money(max(mrr_threshold - c['mrr'], 0.0))}"

    bottom = build_bottom_line(c, currency_symbol)

    summary = f""" 🚀 RUN SUMMARY — {company}
────────────────────────────────────────────
💰 CASH
• Available: {currency_symbol}{money(cash)}
• Monthly Burn (ex-debt): {currency_symbol}{money(burn)}
• Debt-adjusted Burn (incl. debt): {currency_symbol}{money(burn_incl)}
• Static Runway (incl. debt): {base_runway}
• Sources → Bootstrap {currency_symbol}{money(c['bootstrap'])}, Equity {currency_symbol}{money(c['equity'])}, Grants {currency_symbol}{money(c['grant'])}, Loan {currency_symbol}{money(loan)}

📊 TRAJECTORY
• Revenue Growth: +{c['growth_pct']:.1f}% per month
• Cost Growth: +{c['cost_growth_pct']:.1f}% per month
• Operational Breakeven: {op_str}
• Cash Breakeven: {cash_str}
• Funding Gap to Operational BE: {currency_symbol}{money(c['funding_gap'])}
{loan_lines}
🏎️ EFFICIENCY
ARR: {arr_str}
• Burn Multiple: {bm_str}
• MRR Threshold at Op BE: {mrr_threshold_str} (costs + active debt)
• Current MRR: {mrr_current_str}
• MRR Gap: {mrr_gap_str}

────────────────────────────────────────────
{bottom}"""
    return summary

# ----------------- CLI inputs & wiring -----------------

def load_inputs() -> dict:
    print("\n🛫 RunwayToFlight — Founder Runway & Liftoff Planner (14 steps)\n")
    ask = lambda i, t: input(f"[{i}] {t}: ").strip()
    d: dict[str, str] = {}

    print("\n🏢 SECTION — Company")
    d["company_name"]   = ask(1, "Company name (text, e.g. Andiamo Tech)")
    d["formation_date"] = coerce_date(ask(2, "Formation date (YYYY-MM or YYYY-MM-DD)"))

    print("\n💵 SECTION — Revenue & Costs")
    d["mrr"]      = ask(3, "Current MRR (number, USD per month)")
    d["last_mrr"] = ask(4, "Last month MRR (number, USD per month)")
    d["mrc"]      = ask(5, "Total monthly costs MRC (number, USD per month)")

    print("\n📈 SECTION — Trajectory")
    d["growth_pct"]      = ask(6, "Expected MRR growth percent per month (e.g. 10 or 10%)")
    d["cost_growth_pct"] = ask(7, "Cost growth percent per month (e.g. 2 or 2%)")

    print("\n🏗️ SECTION — Capital Sources")
    d["bootstrap_cash"]       = ask(8, "Bootstrap or founder funds (number, USD)")
    d["external_equity_cash"] = ask(9, "Angel and VC funding combined (number, USD)")
    d["grant_cash"]           = ask(10, "Grants or non dilutive funds (number, USD)")

    loan_str = ask(11, "Loans or credit lines (number, USD)")
    d["loan_cash"] = loan_str
    loan_val = to_float(loan_str)

    if loan_val > 0:
        d["loan_apr_pct"]    = ask(12, "Loan APR percent (e.g. 3 or 3%) [optional, default 3%]")
        d["loan_term_years"] = ask(13, "Loan term in years (e.g. 3) [optional, default 3]")
    else:
        d["loan_apr_pct"]    = "0"
        d["loan_term_years"] = "0"

    print("\n🎨 SECTION — Visual")
    d["accent_colors"] = ask(14, "Accent colors (hex, e.g. #12c04c)")
    return d

def save_files(prompt: str, summary: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt)
    (out / "summary.txt").write_text(summary)
    print(f"\n✅ Files saved to {out.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="RunwayToFlight — Founder Runway & Liftoff Planner")
    parser.add_argument("--json", help="Provide inputs as JSON")
    parser.add_argument("--outdir", default="runwaytoflight_out")
    parser.add_argument("--currency", default="$")
    parser.add_argument("--start", help="Optional base date for simulations (YYYY-MM or YYYY-MM-DD)")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in output")
    args = parser.parse_args()

    if args.json:
        inputs = json.loads(args.json)
        if "formation_date" in inputs:
            inputs["formation_date"] = coerce_date(inputs["formation_date"])
        if "accent_colors" not in inputs:
            inputs["accent_colors"] = "#12c04c"
    else:
        inputs = load_inputs()

    calc = compute(inputs, base_date_str=args.start, no_emoji=args.no_emoji)
    prompt = build_prompt(inputs, calc, currency_symbol=args.currency)
    summary = build_summary(calc, currency_symbol=args.currency, no_emoji=args.no_emoji)

    save_files(prompt, summary, args.outdir)

    print("\n=== PROMPT ===\n", prompt)
    print("\n=== SUMMARY ===\n", summary)

if __name__ == "__main__":
    main()
