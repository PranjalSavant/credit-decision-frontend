"""Streamlit frontend. Run:  python -m streamlit run app.py"""
import json, time
from dataclasses import asdict
from datetime import datetime
import pandas as pd
import streamlit as st
from engine import (Policy, ConsentStore, MockBankAPI, PROFILES, clean_transactions,
                    build_features, score, decide, size_loan, annuity_pv, inr)

st.set_page_config(page_title="AI Credit Decisioning", page_icon="🏦", layout="wide")
NAVY, ACCENT, GREEN, AMBER, RED = "#0B2A5B", "#1D5FD1", "#2E8B57", "#D98E04", "#B83232"
BAND_COL = {"Low risk": GREEN, "Medium risk": AMBER, "High risk": RED}

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.stApp {{ background:#E8F0FA; }} .block-container {{ padding-top:1.2rem; max-width:1250px; }}
h1,h2,h3,h4 {{ font-family:'Source Serif 4',serif !important; color:{NAVY}; }}
section[data-testid="stSidebar"] {{ background:#F4F8FD; border-right:1px solid #CFDDF0; }}
.hero {{ background:linear-gradient(120deg,{NAVY} 0%,#1D4E9E 100%); border-radius:18px; padding:26px 32px; margin-bottom:14px; }}
.hero .eyebrow {{ color:#8EC5FF; font-weight:600; letter-spacing:.12em; font-size:.78rem; }}
.hero h1 {{ color:#fff !important; font-size:2.05rem; margin:.3rem 0 .2rem; }}
.hero p {{ color:#C5D8F2; margin:0; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:#fff; padding:6px; border-radius:12px; border:1px solid #CFDDF0; }}
.stTabs [data-baseweb="tab"] {{ border-radius:8px; padding:8px 16px; font-weight:500; }}
.stTabs [aria-selected="true"] {{ background:{NAVY}; color:#fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
.card {{ background:#fff; border:1px solid #CFDDF0; border-radius:14px; padding:18px 20px; height:100%; box-shadow:0 1px 3px rgba(11,42,91,.10); }}
.card.dark {{ background:{NAVY}; border:none; }}
.card .eyebrow {{ color:{ACCENT}; font-weight:600; letter-spacing:.08em; font-size:.75rem; }}
.card.dark .eyebrow, .card.dark h4 {{ color:#8EC5FF !important; }}
.card h4 {{ margin:.25rem 0 .35rem; font-size:1.3rem; }}
.card p {{ margin:0; color:#4A5D78; font-size:.93rem; line-height:1.5; }} .card.dark p {{ color:#D6E4F7; }}
.num {{ display:inline-flex; width:30px; height:30px; border-radius:50%; background:{ACCENT}; color:#fff; align-items:center; justify-content:center; font-weight:600; margin-bottom:6px; }}
.arrow {{ color:#1D5FD1; font-size:1.8rem; text-align:center; padding-top:80px; }}
.kpi .v {{ font-family:'Source Serif 4',serif; font-size:1.9rem; font-weight:700; color:{NAVY}; line-height:1.1; }}
.kpi .l {{ color:#5F7491; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }}
.pill {{ display:inline-block; padding:5px 14px; border-radius:999px; color:#fff; font-weight:600; font-size:.9rem; }}
.promise {{ background:#fff; border-left:5px solid {ACCENT}; padding:12px 18px; border-radius:8px; margin-top:16px; color:{NAVY}; }}
.meter {{ display:flex; height:14px; border-radius:7px; overflow:hidden; position:relative; margin:10px 0 4px; }}
.mark {{ position:absolute; top:-5px; width:4px; height:24px; background:{NAVY}; border-radius:2px; }}
.bar-row {{ margin:10px 0; font-size:.9rem; color:{NAVY}; }}
.bar-track {{ background:#D9E5F5; border-radius:6px; height:12px; margin-top:4px; }}
.bar-fill {{ height:12px; border-radius:6px; }}
.bar-row.bind {{ background:#E4EEFF; border:1px solid #9DBDF0; border-radius:10px; padding:8px 12px; }}
.stepper {{ display:flex; gap:8px; margin:6px 0 14px; }}
.stepper div {{ flex:1; background:#EAF5EE; color:{GREEN}; border-radius:8px; padding:8px; text-align:center; font-size:.82rem; font-weight:600; }}
.foot {{ color:#5F7491; font-size:.82rem; margin-top:14px; }}
div.stButton > button {{ background:{ACCENT}; color:#fff; border:none; border-radius:10px; padding:.65rem 1.4rem; font-weight:600; }}
div.stButton > button:hover {{ background:{NAVY}; color:#fff; }}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("store", ConsentStore()); ss.setdefault("log", []); ss.setdefault("result", None)


def card(eyebrow, title, body, dark=False, num=None):
    n = f'<div class="num">{num}</div>' if num else ""
    return (f'<div class="card {"dark" if dark else ""}">{n}<div class="eyebrow">{eyebrow}</div>'
            f'<h4>{title}</h4><p>{body}</p></div>')


def kpi(label, value, sub=""):
    return (f'<div class="card kpi"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="l" style="text-transform:none">{sub}</div></div>')


def risk_meter(pdef, p):
    lo, me = p.low_pd_max / .5 * 100, (p.medium_pd_max - p.low_pd_max) / .5 * 100
    pos = min(pdef, .5) / .5 * 100
    return (f'<div class="meter"><div style="width:{lo}%;background:{GREEN}"></div>'
            f'<div style="width:{me}%;background:{AMBER}"></div><div style="flex:1;background:{RED}"></div>'
            f'<div class="mark" style="left:calc({pos}% - 2px)"></div></div>'
            f'<div class="l" style="display:flex;justify-content:space-between;color:#5F7491;font-size:.75rem">'
            f'<span>0%</span><span>Low ≤ {p.low_pd_max:.0%}</span><span>Medium ≤ {p.medium_pd_max:.0%}</span><span>50%+ High</span></div>')


def limit_bars(sz):
    mx = max(max(sz["parts"].values()), 1)
    out = ""
    for k, v in sz["parts"].items():
        b = k == sz["binding"]
        out += (f'<div class="bar-row {"bind" if b else ""}"><b>{k}</b>{"  ← lowest, so this applies" if b else ""}'
                f'<span style="float:right;font-weight:600">{inr(v)}</span><div class="bar-track"><div class="bar-fill" '
                f'style="width:{v / mx * 100:.0f}%;background:{ACCENT if b else "#9DB4D3"}"></div></div></div>')
    return out


def reason_bars(reasons):
    mx = max(abs(r["impact"]) for r in reasons) or 1
    out = ""
    for r in reasons:
        col = RED if r["impact"] > 0 else GREEN
        out += (f'<div class="bar-row">{r["factor"]} <span style="float:right;color:{col};font-weight:600">'
                f'{"▲ raises" if r["impact"] > 0 else "▼ lowers"} risk ({r["impact"]:+.2f})</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{abs(r["impact"]) / mx * 100:.0f}%;background:{col}"></div></div></div>')
    return out


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    company = st.text_input("Company name", "[COMPANY NAME]")
    st.markdown("**Bank policy** · illustrative, the bank sets final values")
    pol = Policy(
        emi_limit_pct=st.slider("EMI limit (% of income)", 20, 70, 50, 5) / 100,
        low_pd_max=st.slider("Low risk if default prob. ≤ (%)", 1, 20, 10) / 100,
        medium_pd_max=st.slider("Medium risk if default prob. ≤ (%)", 10, 50, 25) / 100,
        medium_pct_of_eligible=st.slider("Medium risk: % of eligible", 10, 100, 70, 5) / 100,
        low_cap=st.number_input("Low band ceiling (₹)", 0, 10_000_000, 1_000_000, 50_000),
        medium_cap=st.number_input("Medium band ceiling (₹)", 0, 10_000_000, 500_000, 50_000),
        interest_pa=st.slider("Interest rate p.a. (%)", 0, 36, 18) / 100,
        tenure_months=st.select_slider("Tenure (months)", [6, 9, 12, 18, 24, 36], 12),
        gold_ltv=st.slider("Gold loan LTV (%)", 40, 90, 75) / 100)

st.markdown(f'<div class="hero"><div class="eyebrow">{company.upper()} · AI CREDIT DECISIONING</div>'
            '<h1>From your bank\'s data to a lending decision in minutes</h1>'
            '<p>Bureau and Account Aggregator data in, explainable risk decision and safe loan amount out.</p></div>',
            unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Assess a customer", "Policy & sizing", "Benefits & pilot", "Data & audit"])

# ===== Overview =========================================================
with t1:
    cols = st.columns([10, 1, 10, 1, 10, 1, 10])
    steps = [("BANK API", "Collect", "Bureau variables and AA bank data, pulled securely from your bank's systems"),
             ("AI ENGINE", "Analyse", "Cleans transactions, builds risk features and scores default probability"),
             ("RISK", "Decide", "Low, medium or high risk, with clear reasons for the committee"),
             ("AMOUNT", "Size", "The safe loan amount this customer can repay")]
    for i, s in enumerate(steps):
        cols[i * 2].markdown(card(*s, dark=(i == 3), num=i + 1), unsafe_allow_html=True)
        if i < 3: cols[i * 2 + 1].markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)
    st.markdown('<div class="promise"><b>Your data stays yours:</b> we process only on your instructions · '
                'stored in India · deleted when consent expires</div>', unsafe_allow_html=True)
    st.write("")
    k = st.columns(3)
    k[0].markdown(kpi("Turnaround", "Minutes", "not days of paperwork"), unsafe_allow_html=True)
    k[1].markdown(kpi("Data residency", "India", "bank owns its data"), unsafe_allow_html=True)
    k[2].markdown(kpi("Every decision", "Explained", "reasons auditors can see"), unsafe_allow_html=True)
    st.info("Go to **Assess a customer** to run the full flow on a demo profile or your own CSV.")

# ===== Assess ===========================================================
with t2:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 1 · Customer & consent")
            cid = st.text_input("Customer ID", "CUST-1001")
            src = st.selectbox("Data source", list(PROFILES) + ["Upload my own CSV"],
                               help="Demo profiles come from a mock bank API (MockBankAPI in engine.py).")
            up = None
            if src == "Upload my own CSV":
                up = st.file_uploader("AA transactions CSV: date, description, amount, type (Cr/Dr)", type="csv")
            days = st.select_slider("Consent valid for (days)", [30, 60, 90, 180, 365], 90)
            consent = st.checkbox("Customer has consented to fetch bureau + bank (AA) data")
        with st.container(border=True):
            st.markdown("#### 2 · Bureau variables")
            prof = PROFILES.get(src, {}).get("bureau") or dict(score=700, dpd=0, util=0.3, enq=2)
            thin = st.checkbox("No bureau history (thin file)", value=(src == "Thin file"), key=f"thin{src}")
            bureau = None if thin else dict(
                score=st.slider("Bureau score", 300, 900, prof["score"], key=f"s{src}"),
                dpd=st.number_input("Missed payments (24m)", 0, 20, prof["dpd"], key=f"d{src}"),
                util=st.slider("Credit utilisation", 0.0, 1.0, float(prof["util"]), 0.05, key=f"u{src}"),
                enq=st.number_input("Enquiries (6m)", 0, 30, prof["enq"], key=f"e{src}"))
        with st.container(border=True):
            st.markdown("#### 3 · Loan request")
            requested = st.number_input("Amount requested (₹)", 10_000, 10_000_000, 300_000, 10_000)
            gold = st.number_input("Gold collateral value (₹, optional)", 0, 10_000_000, 0, 10_000)
            run = st.button("▶  Run decision", use_container_width=True)

    if run:
        if not consent: st.error("Consent is required. Tick the consent box first.")
        elif src == "Upload my own CSV" and up is None: st.error("Upload a transactions CSV first.")
        else:
            t0 = time.time()
            with st.status("Running credit decision…", expanded=False) as status:
                ss.store.grant(cid, "Credit decisioning", days)
                raw = pd.read_csv(up) if up else None
                if raw is None: _, raw = MockBankAPI().fetch(src)
                ss.store.store(cid, dict(txns=raw))
                clean, cstats = clean_transactions(raw)
                feats = build_features(clean)
                pdef, reasons = score(feats, bureau)
                band, decision = decide(pdef, pol)
                sz = size_loan(feats["monthly_income"], feats["existing_emi"], requested, band, pol, gold)
                secs = time.time() - t0
                status.update(label=f"Decision ready in {secs:.1f}s", state="complete")
            ss.result = dict(customer=cid, band=band, decision=decision, pd=pdef, reasons=reasons, feats=feats,
                             size=sz, clean=clean, cstats=cstats, secs=secs, pol=pol)
            ss.log.append(dict(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), customer=cid, band=band,
                               decision=decision, default_prob=f"{pdef:.1%}", loan=inr(sz["amount"]),
                               policy=json.dumps(asdict(pol))))

    with right:
        r = ss.result
        if not r:
            st.markdown(card("RESULTS", "Ready when you are",
                             "Complete the three sections on the left and press <b>Run decision</b>. "
                             "Your risk band, loan amount and reasons will appear here."), unsafe_allow_html=True)
        else:
            col, f = BAND_COL[r["band"]], r["feats"]
            st.markdown('<div class="stepper"><div>✓ Collect</div><div>✓ Analyse</div><div>✓ Decide</div><div>✓ Size</div></div>',
                        unsafe_allow_html=True)
            st.markdown(f'#### {r["customer"]} &nbsp; <span class="pill" style="background:{col}">{r["band"]} · {r["decision"]}</span>',
                        unsafe_allow_html=True)
            m = st.columns(3)
            m[0].markdown(kpi("Default probability", f"{r['pd']:.1%}"), unsafe_allow_html=True)
            m[1].markdown(kpi("Recommended loan", inr(r["size"]["amount"])), unsafe_allow_html=True)
            m[2].markdown(kpi("Decision time", f"{r['secs']:.1f}s"), unsafe_allow_html=True)
            st.markdown(risk_meter(r["pd"], r["pol"]), unsafe_allow_html=True)
            if r["band"] == "High risk":
                sec = r["size"]["secured_option"]
                st.warning("Unsecured loan rejected. **Secured only (e.g. gold).** " + (
                    f"Secured option up to {inr(sec)} at {r['pol'].gold_ltv:.0%} LTV." if sec
                    else "Enter a gold collateral value to see the secured limit."))
            elif r["band"] == "Medium risk":
                st.info("Refer to the credit committee with the reasons below.")
            with st.container(border=True):
                st.markdown("##### Loan amount = the lowest of")
                st.markdown(limit_bars(r["size"]), unsafe_allow_html=True)
                st.caption(f"Income {inr(f['monthly_income'])} × EMI limit {r['pol'].emi_limit_pct:.0%} = "
                           f"{inr(r['size']['emi_ceiling'])}; existing EMIs {inr(f['existing_emi'])} leave "
                           f"{inr(r['size']['emi_room'])} a month.")
            with st.container(border=True):
                st.markdown("##### Reasons for the committee")
                st.markdown(reason_bars(r["reasons"]), unsafe_allow_html=True)
            with st.expander("Risk features built from bank data"):
                st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
            with st.expander(f"Cleaned transactions ({r['cstats']['removed']} bad rows removed of {r['cstats']['rows_in']})"):
                st.dataframe(r["clean"], use_container_width=True)
            st.download_button("⬇ Download decision (JSON)", json.dumps(
                dict(customer=r["customer"], band=r["band"], decision=r["decision"], default_prob=r["pd"],
                     loan=r["size"]["amount"], binding=r["size"]["binding"], reasons=r["reasons"],
                     policy=asdict(r["pol"])), indent=2, default=str), f"decision_{r['customer']}.json")

# ===== Policy ===========================================================
with t3:
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown("### Risk bands")
        st.table(pd.DataFrame([
            ["Low risk", "Approve", "Full eligible amount"],
            ["Medium risk", "Committee review", f"Up to {pol.medium_pct_of_eligible:.0%} of eligible"],
            ["High risk", "Reject", "Secured only (e.g. gold)"]], columns=["Band", "Decision", "Loan limit"]).set_index("Band"))
        st.caption(f"Default probability: Low ≤ {pol.low_pd_max:.0%} · Medium ≤ {pol.medium_pd_max:.0%} · High above.")
    with b:
        st.markdown(card("SIZING RULE", "Loan amount = the lowest of",
                         "1. Repayment capacity from AA income and existing EMIs<br>2. Cap for the risk band<br>"
                         "3. Amount requested", dark=True), unsafe_allow_html=True)
    st.markdown("### Worked example")
    e = st.columns(3)
    inc = e[0].number_input("Monthly income (₹)", 0, 1_000_000, 40_000, 1000)
    ex = e[1].number_input("Existing EMIs (₹)", 0, 1_000_000, 6_000, 500)
    mo = e[2].number_input("Months", 3, 60, 12, 3)
    lim, room = inc * pol.emi_limit_pct, max(0, inc * pol.emi_limit_pct - ex)
    st.success(f"Income {inr(inc)}, EMI limit {pol.emi_limit_pct:.0%} = {inr(lim)}. Existing EMIs {inr(ex)} leave "
               f"{inr(room)} a month, about **{inr(annuity_pv(room, pol.interest_pa, mo))}** over {mo} months "
               f"(at {pol.interest_pa:.0%} p.a.).")
    st.markdown('<div class="foot">Illustrative policy · your bank sets the final cut-offs, EMI limit and caps</div>',
                unsafe_allow_html=True)

# ===== Benefits & pilot =================================================
with t4:
    ben = [("Fewer bad loans", "Spot risk early using bureau and real bank data"),
           ("Faster approvals", "Decisions in minutes instead of days of paperwork"),
           ("Right-sized loans", "Lend what each customer can actually repay"),
           ("Consistent decisions", "Same rules in every branch, with reasons auditors can see"),
           ("More customers", "Safely serve thin-file borrowers using bank data"),
           ("Compliance-ready", "Consent logs, data in India, the bank owns its data")]
    for row in (ben[:3], ben[3:]):
        for c, (t, d) in zip(st.columns(3), row):
            c.markdown(card("", t, d, dark=True), unsafe_allow_html=True)
        st.write("")
    st.markdown("### Next step: 3-month pilot")
    p1, p2 = st.columns(2)
    nb = p1.number_input("Number of pilot branches", 1, 500, 5)
    kpis = p2.multiselect("Agreed success measures", ["Early-default (bad loan) rate", "Decision turnaround time",
                          "Approval rate", "Thin-file customers served", "Committee override rate", "Audit exceptions"],
                          ["Early-default (bad loan) rate", "Decision turnaround time"])
    st.success(f"**Next step:** a 3-month pilot in **{nb}** branches, with agreed success measures: {', '.join(kpis)}")
    st.download_button("⬇ Download pilot summary",
                       f"{company}: 3-month pilot in {nb} branches. Success measures: " + "; ".join(kpis), "pilot_plan.txt")

# ===== Data & audit =====================================================
with t5:
    st.markdown('<div class="promise"><b>Your data stays yours:</b> processed only when you press Run · stored in '
                'India · deleted when consent expires · the bank owns its data</div>', unsafe_allow_html=True)
    st.markdown("### Consent log")
    cl = pd.DataFrame(ss.store.consents)
    st.dataframe(cl if len(cl) else pd.DataFrame(columns=["customer_id", "expires_at", "status"]), use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Delete data for expired consents", use_container_width=True):
        st.toast(f"Deleted data for {ss.store.purge_expired()} expired consent(s)")
    if c2.button("Simulate consent expiry (demo)", use_container_width=True):
        for c in ss.store.consents: c["expires_at"] = datetime.now()
        st.toast(f"Deleted data for {ss.store.purge_expired()} customer(s)")
    st.markdown("### Decision log")
    st.dataframe(pd.DataFrame(ss.log), use_container_width=True)
