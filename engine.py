"""Credit decisioning engine: Collect -> Analyse -> Decide -> Size.

Everything here is plain Python (no UI). To go live, replace MockBankAPI with a
class that calls the bank's real bureau / Account Aggregator (AA) endpoints,
and replace the hand-set weights in `score()` with a trained model.
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- policy
@dataclass
class Policy:
    """Illustrative policy: the bank sets the final cut-offs, EMI limit and caps."""
    emi_limit_pct: float = 0.50          # max share of income going to EMIs
    low_pd_max: float = 0.10             # PD <= this  -> Low risk
    medium_pd_max: float = 0.25          # PD <= this  -> Medium, above -> High
    medium_pct_of_eligible: float = 0.70 # Medium risk limit
    low_cap: int = 1_000_000             # product ceiling, Low band (Rs)
    medium_cap: int = 500_000            # product ceiling, Medium band (Rs)
    interest_pa: float = 0.18            # used to turn monthly capacity into a loan amount
    tenure_months: int = 12
    gold_ltv: float = 0.75               # secured option for High risk


def inr(x: float) -> str:
    s = str(int(round(abs(x))))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return ("-" if x < 0 else "") + "₹" + s


# ---------------------------------------------------------------- consent & data store
class ConsentStore:
    """Consent log + data held per customer. Data is deleted when consent expires."""
    REGION = "India (data residency)"

    def __init__(self):
        self.consents, self.data = [], {}

    def grant(self, customer_id, purpose, days):
        now = datetime.now()
        self.consents.append(dict(customer_id=customer_id, purpose=purpose, granted_at=now,
                                  expires_at=now + timedelta(days=days), status="ACTIVE",
                                  stored_in=self.REGION))

    def store(self, customer_id, payload):
        self.data[customer_id] = payload

    def purge_expired(self, now=None):
        now, n = now or datetime.now(), 0
        for c in self.consents:
            if c["status"] == "ACTIVE" and c["expires_at"] <= now:
                c["status"] = "EXPIRED - DATA DELETED"
                n += self.data.pop(c["customer_id"], None) is not None
        return n


# ---------------------------------------------------------------- 01 COLLECT (bank API stub)
PROFILES = {
    "Low risk":    dict(income=60000, emi=6000,  bounces=0, vol=0.02, bureau=dict(score=790, dpd=0, util=0.25, enq=1)),
    "Medium risk": dict(income=40000, emi=6000,  bounces=1, vol=0.15, bureau=dict(score=690, dpd=1, util=0.55, enq=3)),
    "High risk":   dict(income=28000, emi=11000, bounces=4, vol=0.40, bureau=dict(score=580, dpd=3, util=0.90, enq=7)),
    "Thin file":   dict(income=35000, emi=0,     bounces=0, vol=0.08, bureau=None),
}


class MockBankAPI:
    """Stand-in for the bank's bureau + AA APIs. Returns (bureau dict | None, raw txn DataFrame)."""
    def fetch(self, profile: str, seed: int = 7):
        p, rng, rows = PROFILES[profile], np.random.default_rng(seed), []
        base = datetime.today().replace(day=1)
        for m in range(6, 0, -1):
            t0 = (base - pd.DateOffset(months=m)).to_pydatetime()
            rows.append((t0, "Salary credit ACME", p["income"] * (1 + rng.normal(0, p["vol"])), "Cr"))
            if p["emi"]:
                rows.append((t0 + timedelta(days=4), "NACH EMI home loan", p["emi"], "Dr"))
            for _ in range(8):
                rows.append((t0 + timedelta(days=int(rng.integers(2, 27))),
                             str(rng.choice(["UPI grocery", "UPI rent", "POS fuel", "Electricity bill", "UPI shopping"])),
                             float(rng.uniform(0.04, 0.10) * p["income"]), "Dr"))
            if m <= p["bounces"]:
                rows.append((t0 + timedelta(days=6), "EMI BOUNCE insufficient funds", 500, "Dr"))
        df = pd.DataFrame(rows, columns=["date", "description", "amount", "type"])
        dirty = pd.DataFrame([(df.date[0], df.description[0], df.amount[0], "Cr"),   # duplicate
                              ("not-a-date", "corrupt row", 100, "Dr"),               # bad date
                              (df.date[1], "blank amount", None, "Dr")],              # missing amount
                             columns=df.columns)
        return p["bureau"], pd.concat([df, dirty], ignore_index=True)


# ---------------------------------------------------------------- 02 ANALYSE
def clean_transactions(df: pd.DataFrame):
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    n0 = len(d)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").abs()
    d["type"] = d["type"].astype(str).str.strip().str.upper().str[0].map({"C": "CREDIT", "D": "DEBIT"})
    d["description"] = d["description"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    d = d.dropna(subset=["date", "amount", "type"])
    d = d[d.amount > 0].drop_duplicates(["date", "description", "amount", "type"])
    d = d.sort_values("date").reset_index(drop=True)
    return d, dict(rows_in=n0, rows_out=len(d), removed=n0 - len(d))


def build_features(d: pd.DataFrame) -> dict:
    d = d.assign(month=d.date.dt.to_period("M"))
    cr, db = d[d.type == "CREDIT"], d[d.type == "DEBIT"]
    sal = cr[cr.description.str.contains("SALARY|PAYROLL|WAGES")]
    monthly_inc = (sal if len(sal) >= 2 else cr).groupby("month").amount.sum()
    income = float(monthly_inc.mean())
    emi = db[db.description.str.contains(r"\bEMI\b|LOAN|NACH|ECS") & ~db.description.str.contains("BOUNCE")]
    existing = float(emi.groupby("month").amount.sum().mean()) if len(emi) else 0.0
    return dict(
        months_of_data=int(d.month.nunique()),
        monthly_income=income,
        existing_emi=existing,
        emi_burden=existing / income if income else 1.0,
        volatility=float(monthly_inc.std(ddof=0) / income) if income else 1.0,
        bounces=int(d.description.str.contains("BOUNCE|RETURN|INSUFFICIENT").sum()),
        savings_ratio=float((cr.amount.sum() - db.amount.sum()) / max(cr.amount.sum(), 1)),
    )


def score(f: dict, bureau: dict | None):
    """Transparent log-odds model. Each term is a reason the committee can read."""
    c = {}
    if bureau:
        c["Bureau score"] = -(bureau["score"] - 700) / 100 * 0.9
        c["Missed payments (DPD)"] = 0.5 * bureau["dpd"]
        c["Credit utilisation"] = 1.5 * (bureau["util"] - 0.3)
        c["Recent enquiries"] = 0.15 * max(0, bureau["enq"] - 2)
    else:
        c["Thin file: scored on bank data"] = 0.3
    c["Bounced / returned payments"] = 0.4 * f["bounces"]
    c["Income volatility"] = 1.5 * (f["volatility"] - 0.2)
    c["Existing EMI burden"] = 3.0 * (f["emi_burden"] - 0.25)
    c["Savings ratio"] = -1.0 * (f["savings_ratio"] - 0.10)
    z = -2.2 + sum(c.values())
    pdef = 1 / (1 + math.exp(-z))
    reasons = [dict(factor=k, impact=round(v, 2),
                    effect="raises risk" if v > 0 else "lowers risk")
               for k, v in sorted(c.items(), key=lambda kv: -abs(kv[1]))]
    return pdef, reasons


# ---------------------------------------------------------------- 03 DECIDE
def decide(pdef: float, p: Policy):
    if pdef <= p.low_pd_max:
        return "Low risk", "Approve"
    if pdef <= p.medium_pd_max:
        return "Medium risk", "Committee review"
    return "High risk", "Reject"


# ---------------------------------------------------------------- 04 SIZE
def annuity_pv(emi: float, rate_pa: float, n: int) -> float:
    r = rate_pa / 12
    return emi * n if r == 0 else emi * (1 - (1 + r) ** -n) / r


def size_loan(income, existing_emi, requested, band, p: Policy, collateral=0.0):
    """Loan amount = lowest of: repayment capacity, cap for the risk band, amount requested."""
    room = max(0.0, income * p.emi_limit_pct - existing_emi)
    capacity = annuity_pv(room, p.interest_pa, p.tenure_months)
    band_cap = {"Low risk": min(capacity, p.low_cap),
                "Medium risk": min(capacity * p.medium_pct_of_eligible, p.medium_cap),
                "High risk": 0.0}[band]
    parts = {"Repayment capacity": capacity, "Cap for the risk band": band_cap, "Amount requested": float(requested)}
    binding = min(parts, key=parts.get)
    out = dict(emi_ceiling=income * p.emi_limit_pct, emi_room=room, parts=parts,
               binding=binding, amount=round(min(parts.values()), -2), secured_option=0.0)
    if band == "High risk":
        out["secured_option"] = round(min(requested, collateral * p.gold_ltv), -2)
    return out
