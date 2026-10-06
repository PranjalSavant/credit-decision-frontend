# AI Credit Decisioning: Frontend (Streamlit)

Run: `pip install -r requirements.txt && streamlit run app.py`

## Files
- `app.py`    UI: 5 tabs (How it works, Assess a customer, Policy & sizing, Benefits & pilot, Data & audit)
- `engine.py` Logic: consent store, bank API stub, cleaning, features, scoring, risk band, loan sizing

## PDF requirement -> where implemented
| Requirement | Location |
|---|---|
| 01 Collect bureau + AA data via bank API | `MockBankAPI.fetch` (swap for real API) |
| 02 Clean transactions, risk features, default probability | `clean_transactions`, `build_features`, `score` |
| 03 Low/Medium/High with reasons for committee | `decide`, `score` reasons, "Reasons" panel |
| 04 Safe loan = lowest of capacity, band cap, requested | `size_loan` |
| Bands: Approve / Committee review (70%) / Reject, secured only (gold) | `Policy`, `decide`, gold LTV option |
| EMI limit, cut-offs, caps set by bank | Sidebar -> `Policy` |
| Data: process on instruction, stored in India, deleted on consent expiry | `ConsentStore` + Data & audit tab |
| 6 benefits, 3-month pilot in [__] branches | Benefits & pilot tab |

## Handover notes
- Demo data is synthetic. Scoring weights in `score()` are hand-set placeholders: replace with the trained model.
- State is in-memory (`st.session_state`); add a real database in India for production.
- Sizing uses an annuity at the policy interest rate (18%, 12m), which reproduces the deck's "about 1.5 lakh" for Rs 14,000/month. Without interest it would be Rs 1.68 lakh.
