import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Housing Price Analysis", page_icon="🏡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1 { color: #1a3a6b; }
    h2 { color: #1a3a6b; border-bottom: 2px solid #e0e8f5; padding-bottom: 6px; }
    .stMetric {
        background-color: #f0f5ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2e67b1;
    }
    .equation-box {
        background: linear-gradient(135deg, #f0f5ff, #e8f0fe);
        border-left: 5px solid #1a73e8;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        line-height: 1.8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title('🏡 DSA 250: Housing Price Analysis')
st.subheader('Delilah Hollander')
st.divider()

# ── Data loading & modeling ────────────────────────────────────────────────────
@st.cache_data
def load_and_model():
    df = pd.read_csv('train.csv')
    key_vars = ['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
    df_clean = df[key_vars].dropna()

    feature_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
    X = df_clean[feature_cols]
    y = df_clean['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Simple model (GrLivArea only)
    m1 = LinearRegression().fit(X_train[['GrLivArea']], y_train)
    y_pred1 = m1.predict(X_test[['GrLivArea']])

    # Multiple regression
    m2 = LinearRegression().fit(X_train, y_train)
    y_pred2 = m2.predict(X_test)

    metrics = {
        'simple': {
            'r2':   r2_score(y_test, y_pred1),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred1)),
            'mae':  mean_absolute_error(y_test, y_pred1),
        },
        'multi': {
            'r2':   r2_score(y_test, y_pred2),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred2)),
            'mae':  mean_absolute_error(y_test, y_pred2),
        },
    }

    return df_clean, feature_cols, m1, m2, X_test, y_test, y_pred1, y_pred2, metrics

try:
    df_clean, feature_cols, m1, m2, X_test, y_test, y_pred1, y_pred2, metrics = load_and_model()
except Exception as e:
    st.error(f"Error loading data or building model: {e}")
    st.stop()

corr = df_clean[['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']].corr()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("📋 Project Menu")
page = st.sidebar.radio("Jump to Section:", [
    "📊 Overall Summary",
    "🗂 Data Background",
    "🔍 Detailed EDA",
    "📈 Regression Analysis",
    "🏠 Price Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overall Summary
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overall Summary":
    st.header("1. Project Summary")
    st.write("""
    This project analyzes **1,460 home sales in Ames, Iowa** to find what actually makes a house expensive.
    I examined square footage, year built, bathrooms, and bedrooms — first with exploratory analysis,
    then with a **multiple linear regression model** to quantify each variable's true impact on price.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Homes", f"{len(df_clean):,}")
    with col2:
        st.metric("Avg Sale Price", f"${df_clean['SalePrice'].mean():,.0f}")
    with col3:
        st.metric("Avg Size", f"{df_clean['GrLivArea'].mean():,.0f} sq ft")
    with col4:
        st.metric("Model R²", f"{metrics['multi']['r2']:.3f}")

    st.subheader("Key Findings")
    st.success(f"✅ **Square footage** is the single strongest predictor (r = {corr.loc['SalePrice','GrLivArea']:.3f})")
    st.success(f"✅ **Multiple regression** explains **{metrics['multi']['r2']*100:.1f}%** of price variation (R² = {metrics['multi']['r2']:.3f})")
    st.info(f"📐 The model's average prediction error is **±${metrics['multi']['rmse']:,.0f}** (RMSE)")
    st.info(f"🛁 Each additional full bath adds ≈ **${m2.coef_[feature_cols.index('FullBath')]:,.0f}** after controlling for size")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Background
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂 Data Background":
    st.header("2. Data Source & Preparation")
    st.write("**Source:** Ames Housing Dataset (Kaggle house-prices-advanced-regression-techniques)")
    st.write(f"The original dataset has **{80}+ columns**. I narrowed it to 5 key variables for a focused analysis.")

    st.subheader("Variables Selected")
    var_info = {
        'Variable': ['SalePrice', 'GrLivArea', 'FullBath', 'YearBuilt', 'BedroomAbvGr'],
        'Role': ['Target (Y)', 'Predictor', 'Predictor', 'Predictor', 'Predictor'],
        'Description': [
            'Final sale price in dollars',
            'Above-grade living area square footage',
            'Number of full bathrooms',
            'Year the house was originally built',
            'Number of bedrooms above basement level',
        ],
        'Type': ['Continuous', 'Continuous', 'Ordinal', 'Continuous', 'Ordinal'],
    }
    st.dataframe(pd.DataFrame(var_info), use_container_width=True)

    st.subheader("Data Cleaning Code")
    st.code("""
key_vars = ['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
df_clean = df[key_vars].dropna()   # 0 rows dropped — all 1,460 records complete
    """, language='python')

    st.subheader("Descriptive Statistics")
    st.dataframe(df_clean.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Detailed EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detailed EDA":
    st.header("3. Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🌡️ Correlation Heatmap", "🔵 Scatter Plots"])

    with tab1:
        st.write("Sale Price and Square Footage are both **right-skewed** — most homes cluster at lower values with a long tail of expensive/large properties.")
        try:
            st.image("01_distributions.png", use_container_width=True)
        except:
            st.error("Run `DSA250_Housing_Project_Starter.py` to generate chart images.")

    with tab2:
        st.write("**GrLivArea** and **FullBath** have the strongest correlations with SalePrice.")
        try:
            st.image("02_correlation_matrix.png", use_container_width=True)
        except:
            st.error("Correlation matrix image missing. Run the analysis script.")

        st.subheader("Pearson Correlation with SalePrice")
        price_corr = corr['SalePrice'].drop('SalePrice').sort_values(ascending=False).reset_index()
        price_corr.columns = ['Variable', 'Correlation']
        price_corr['Strength'] = price_corr['Correlation'].apply(
            lambda r: '🔴 Strong' if abs(r) >= 0.5 else '🟡 Moderate' if abs(r) >= 0.3 else '🟢 Weak')
        st.dataframe(price_corr, use_container_width=True)

    with tab3:
        st.write("Each scatter plot encodes a **3rd variable as color**, revealing multi-dimensional patterns at once.")
        try:
            st.image("03_scatterplots.png", use_container_width=True)
        except:
            st.error("Scatterplot image missing. Run the analysis script.")

        st.markdown("""
        **What the colors reveal:**
        - **Top-left (LivArea vs Price, color=FullBath):** Larger homes tend to have more bathrooms AND cost more — all three move together.
        - **Top-right (FullBath vs Price, color=GrLivArea):** High-bath, high-price homes are also the largest (dark purple = big sq footage).
        - **Bottom-left (YearBuilt vs Price, color=Bedrooms):** Newer homes command higher prices regardless of bedroom count.
        - **Bottom-right (Bedrooms vs Price, color=YearBuilt):** 3-bedroom homes are most common; newer ones (yellow) fetch higher prices.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Regression Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Analysis":
    st.header("4. Linear Regression Analysis")

    st.write("""
    I fit two models: a **simple regression** (GrLivArea only) as a baseline,
    and a **multiple regression** using all four predictors. The test set holds out
    20% of the data (292 homes) that the model never saw during training.
    """)

    # Model comparison metrics
    st.subheader("Model Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📉 Simple Regression** (GrLivArea only)")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['simple']['r2']:.3f}")
        c2.metric("RMSE", f"${metrics['simple']['rmse']:,.0f}")
        c3.metric("MAE", f"${metrics['simple']['mae']:,.0f}")

    with col2:
        st.markdown("**📈 Multiple Regression** (all 4 features)")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['multi']['r2']:.3f}",
                  delta=f"+{metrics['multi']['r2'] - metrics['simple']['r2']:.3f} vs simple")
        c2.metric("RMSE", f"${metrics['multi']['rmse']:,.0f}",
                  delta=f"-${metrics['simple']['rmse'] - metrics['multi']['rmse']:,.0f} vs simple")
        c3.metric("MAE", f"${metrics['multi']['mae']:,.0f}")

    # Model equation
    st.subheader("Model Equation")
    bed_coef  = m2.coef_[feature_cols.index('BedroomAbvGr')]
    bath_coef = m2.coef_[feature_cols.index('FullBath')]
    year_coef = m2.coef_[feature_cols.index('YearBuilt')]
    liv_coef  = m2.coef_[feature_cols.index('GrLivArea')]

    st.markdown(f"""
    <div class="equation-box">
    SalePrice = ${m2.intercept_:,.0f}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ <b>${liv_coef:,.2f}</b> × GrLivArea<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ <b>${bed_coef:,.2f}</b> × BedroomAbvGr<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ <b>${bath_coef:,.2f}</b> × FullBath<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ <b>${year_coef:,.2f}</b> × YearBuilt
    </div>
    """, unsafe_allow_html=True)

    st.caption("""
    **Interpreting the coefficients:** Controlling for other variables — each sq ft adds ~${:.0f},
    each full bath adds ~${:.0f}, and each year newer adds ~${:.0f}.
    The negative bedroom coefficient reflects *multicollinearity*: once sq footage is accounted for,
    more bedrooms in the same space actually correlates with smaller individual rooms.
    """.format(liv_coef, bath_coef, year_coef))

    # Coefficient table
    st.subheader("Coefficient Table")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient ($)': [f"${c:,.2f}" for c in m2.coef_],
        'Abs Impact': np.abs(m2.coef_),
        'Interpretation': [
            f"${liv_coef:,.0f} added per sq ft",
            f"${bed_coef:,.0f} per bedroom (controlled)",
            f"${bath_coef:,.0f} per full bath (controlled)",
            f"${year_coef:,.0f} per year newer (controlled)",
        ]
    }).sort_values('Abs Impact', ascending=False)
    st.dataframe(coef_df[['Feature', 'Coefficient ($)', 'Interpretation']], use_container_width=True)

    # Regression diagnostic images
    st.subheader("Regression Diagnostics")
    try:
        st.image("04_regression_analysis.png", use_container_width=True)
    except:
        st.error("Run the analysis script to generate `04_regression_analysis.png`.")

    # Interactive predicted vs actual (live chart)
    st.subheader("Interactive: Actual vs. Predicted (test set)")
    fig, ax = plt.subplots(figsize=(7, 5))
    max_val = max(y_test.max(), y_pred2.max())
    ax.scatter(y_test, y_pred2, alpha=0.45, s=18, color='#5c6bc0')
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.8, label='Perfect prediction')
    ax.set_xlabel('Actual SalePrice ($)')
    ax.set_ylabel('Predicted SalePrice ($)')
    ax.set_title(f'Multiple Regression — Predicted vs. Actual (R²={metrics["multi"]["r2"]:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Sample predictions table
    st.subheader("Sample Predictions")
    sample_homes = pd.DataFrame({
        'GrLivArea':    [1000, 1500, 2000, 2500, 3000],
        'BedroomAbvGr': [2,    3,    3,    4,    4   ],
        'FullBath':     [1,    1,    2,    2,    3   ],
        'YearBuilt':    [1960, 1990, 2000, 2005, 2010],
    })
    sample_preds = m2.predict(sample_homes)
    sample_homes['Predicted Price'] = [f"${p:,.0f}" for p in sample_preds]
    sample_homes['±1σ Range'] = [
        f"${max(0, p - metrics['multi']['rmse']):,.0f} – ${p + metrics['multi']['rmse']:,.0f}"
        for p in sample_preds
    ]
    st.dataframe(sample_homes, use_container_width=True)
    st.caption(f"Uncertainty band = ±1 RMSE (${metrics['multi']['rmse']:,.0f})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Price Predictor (live model)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Price Predictor":
    st.header("5. Real-Time Price Predictor")
    st.write("""
    Adjust the sliders below. The prediction uses the **trained multiple regression model**
    (not a hand-coded formula) — R² = {:.3f}, RMSE ±${:,.0f}.
    """.format(metrics['multi']['r2'], metrics['multi']['rmse']))

    col1, col2 = st.columns(2)
    with col1:
        sqft  = st.slider("🏠 Living Area (sq ft)", 400, 5000, 1500, step=50)
        year  = st.slider("🗓️ Year Built", 1880, 2010, 1990)
    with col2:
        baths = st.slider("🛁 Full Bathrooms", 0, 4, 2)
        beds  = st.slider("🛏️ Bedrooms Above Grade", 0, 8, 3)

    user_home = pd.DataFrame({
        'GrLivArea':    [sqft],
        'BedroomAbvGr': [beds],
        'FullBath':     [baths],
        'YearBuilt':    [year],
    })
    predicted = m2.predict(user_home)[0]
    low  = max(0, predicted - metrics['multi']['rmse'])
    high = predicted + metrics['multi']['rmse']

    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("📌 Predicted Sale Price", f"${predicted:,.0f}")
        st.caption(f"Likely range: ${low:,.0f} – ${high:,.0f}")
    with res_col2:
        # Mini breakdown bar chart
        contrib = {
            'GrLivArea':    liv_coef  * sqft,
            'YearBuilt':    year_coef * year,
            'FullBath':     bath_coef * baths,
            'BedroomAbvGr': bed_coef  * beds,
            'Intercept':    m2.intercept_,
        }
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        colors = ['#42a5f5' if v >= 0 else '#ef5350' for v in contrib.values()]
        ax2.barh(list(contrib.keys()), list(contrib.values()), color=colors, edgecolor='white')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Contribution to predicted price ($)')
        ax2.set_title('What drives this prediction?')
        ax2.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.divider()
    st.subheader("How does this home compare to the dataset?")
    pct_price = (predicted - df_clean['SalePrice'].mean()) / df_clean['SalePrice'].std()
    pct_size  = (sqft - df_clean['GrLivArea'].mean()) / df_clean['GrLivArea'].std()

    if pct_price > 1:
        st.info(f"💰 This home is **above average** in price ({pct_price:.1f} standard deviations above the mean of ${df_clean['SalePrice'].mean():,.0f})")
    elif pct_price < -1:
        st.info(f"💸 This home is **below average** in price ({abs(pct_price):.1f} std devs below the mean)")
    else:
        st.info(f"📊 This home is **near the average** price (within 1 standard deviation of ${df_clean['SalePrice'].mean():,.0f})")
