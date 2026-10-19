# 💰 QuickLoan Risk Dashboard

A Streamlit dashboard for the QuickLoan ensemble-learning challenge. It uses the champion model from the notebook: a **Stacking ensemble** (Random Forest + Gradient Boosting + XGBoost + Logistic Regression, with a Logistic Regression meta-model) that **rejects when P(default) ≥ 0.14**.

## What it shows
1. **Loan decision.** Enter an applicant's details to get **APPROVE / REJECT**, the probability of default, and the expected value of approving.
2. **Portfolio view.** On the held-out test split (1,500 loans), it shows **money saved compared with approving everyone**, **defaulters caught**, and **good borrowers wrongly rejected**.

Expected portfolio results: **AED 6,113,140 saved · 102 / 188 defaulters caught · 131 good borrowers rejected**.

## Files
| File | Purpose |
|---|---|
| `QuickLoan_BakeOff.ipynb` | The notebook: EDA → single models → Voting, Bagging, Boosting, Stacking → recall tuning → leaderboard → verdict |
| `app.py` | The dashboard |
| `loans.csv` | Data (required) |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme and server settings |
| `loan_model.joblib` | *Optional.* The champion saved from the notebook |

If `loan_model.joblib` is missing or can't be loaded, the app trains the same champion from `loans.csv` the first time it starts (about 30–60 s, then cached). The results are identical.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push these files to a GitHub repo, keeping `app.py`, `loans.csv` and `requirements.txt` in the same folder.
2. At share.streamlit.io, choose **Create app**, then pick the repo, branch `main` and main file `app.py`.
3. The first build installs the packages and the first visit trains the model, so allow a couple of minutes. Later visits are fast.
