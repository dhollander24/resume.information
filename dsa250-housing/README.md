# DSA 250 — Housing Price Analysis

**By:** Delilah Hollander | **Course:** DSA 250 — Data Science & Analytics

---

## 🚀 Launch the Interactive App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dhollander24-dsa250-housing-app.streamlit.app)

> **Deploy steps (one-time):** Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → New App → repo: `dhollander24/dsa250-housing`, branch: `main`, file: `app.py` → Deploy.  
> Your badge URL will be `https://<your-app-slug>.streamlit.app`.

[![Run Analysis](https://github.com/dhollander24/dsa250-housing/actions/workflows/run_analysis.yml/badge.svg)](https://github.com/dhollander24/dsa250-housing/actions/workflows/run_analysis.yml)

---

## Project Overview

Analysis of 1,460 home sales in Ames, Iowa to identify the strongest predictors of sale price using **exploratory data analysis** and a **multiple linear regression model** (R² = 0.706).

### Research Questions

1. How strongly does above-ground square footage predict home sale price?
2. Do homes with more full bathrooms sell for more?
3. Are newer homes priced higher than older homes?
4. Does bedroom count predict price once square footage is accounted for?

### Key Findings

| Variable | Correlation (r) | Regression Coef |
|---|---|---|
| GrLivArea (sq ft) | **0.709** | +$107 per sq ft |
| FullBath | 0.561 | controlled |
| YearBuilt | 0.523 | +$960 per year newer |
| BedroomAbvGr | 0.168 | controlled (multicollinear) |

**Multiple regression R² = 0.706 — model explains 70.6% of price variance.**

---

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit interactive dashboard (5 pages) |
| `DSA250_Housing_Project_Starter.py` | Full analysis + regression script |
| `train.csv` | Ames Housing Dataset (1,460 records) |
| `requirements.txt` | Python dependencies |
| `01_distributions.png` | Histograms of key variables |
| `02_correlation_matrix.png` | Pearson correlation heatmap |
| `03_scatterplots.png` | Color-encoded 3-variable scatter plots |
| `04_regression_analysis.png` | Regression diagnostics panel |

## GitHub Actions

The **Run Analysis** workflow triggers automatically when the analysis script or data changes — it regenerates all chart PNGs and commits them back. You can also trigger it manually from the [Actions tab](https://github.com/dhollander24/dsa250-housing/actions).

## Project Website

[View static project page](https://dhollander24.github.io/dsa250-housing/)