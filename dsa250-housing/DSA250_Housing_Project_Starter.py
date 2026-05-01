# DSA 250 Project - Housing Price Analysis
# Delilah Hollander
# This script loads the housing data and runs the main EDA + hypothesis testing + regression modeling

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — required for GitHub Actions / CI (no display needed)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Section 1: Load the data ---

print("Loading data...")

# Load the dataset
# NOTE: Download from Kaggle at: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# The main file is train.csv (contains SalePrice) and test.csv (no SalePrice)

df = pd.read_csv('train.csv')  # Adjust path as needed

# Display basic information
print(f"\nDataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Key variables for this analysis
key_variables = ['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']

print("\nKey Variables for Analysis:")
print(key_variables)

# --- Section 2: Check for missing values ---

print("\nChecking for missing values...")

# Check for missing values in key variables
print("\nMissing Values in Key Variables:")
missing_data = df[key_variables].isnull().sum()
print(missing_data)

# Calculate percentage of missing values
print("\nPercentage Missing:")
print((missing_data / len(df) * 100).round(2))

# Handling missing values (if any)
df_clean = df[key_variables].dropna()
print(f"\nAfter removing rows with missing values: {df_clean.shape[0]} rows remain")

# --- Section 3: Descriptive stats ---

print("\nDescriptive Statistics:")

print("\nSummary Statistics for Key Variables:")
print(df_clean.describe().round(2))

# Additional statistics
print("\nAdditional Statistics:")
for var in key_variables:
    print(f"\n{var}:")
    print(f"  Skewness: {df_clean[var].skew():.3f}")
    print(f"  Kurtosis: {df_clean[var].kurtosis():.3f}")

# --- Section 4: Plot distributions ---

print("\nGenerating distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold')

axes[0, 0].hist(df_clean['SalePrice'], bins=50, edgecolor='black', color='steelblue')
axes[0, 0].set_title('SalePrice Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(df_clean['GrLivArea'], bins=50, edgecolor='black', color='forestgreen')
axes[0, 1].set_title('GrLivArea (Sq Ft) Distribution')
axes[0, 1].set_xlabel('Square Feet')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].hist(df_clean['YearBuilt'], bins=50, edgecolor='black', color='coral')
axes[0, 2].set_title('YearBuilt Distribution')
axes[0, 2].set_xlabel('Year')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(alpha=0.3)

bedroom_counts = df_clean['BedroomAbvGr'].value_counts().sort_index()
axes[1, 0].bar(bedroom_counts.index, bedroom_counts.values, edgecolor='black', color='mediumpurple')
axes[1, 0].set_title('BedroomAbvGr Distribution')
axes[1, 0].set_xlabel('Number of Bedrooms')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(alpha=0.3, axis='y')

bath_counts = df_clean['FullBath'].value_counts().sort_index()
axes[1, 1].bar(bath_counts.index, bath_counts.values, edgecolor='black', color='gold')
axes[1, 1].set_title('FullBath Distribution')
axes[1, 1].set_xlabel('Number of Full Bathrooms')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(alpha=0.3, axis='y')

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('01_distributions.png', dpi=300, bbox_inches='tight')
print("\nDistribution plot saved as '01_distributions.png'")

# --- Section 5: Correlation analysis ---

print("\nCalculating correlations...")

correlation_matrix = df_clean[key_variables].corr()
print("\nPearson Correlation Matrix:")
print(correlation_matrix.round(3))

print("\nCorrelation with SalePrice (Dependent Variable):")
price_correlations = correlation_matrix['SalePrice'].sort_values(ascending=False)
print(price_correlations.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
plt.title('Correlation Matrix - Housing Price Predictors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\nCorrelation matrix saved as '02_correlation_matrix.png'")

# --- Section 6: Enhanced Scatterplots (with 3rd variable color encoding) ---

print("\nGenerating enhanced scatterplots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sale Price vs Independent Variables\n(color = 3rd variable)', fontsize=16, fontweight='bold')

# --- H1: SalePrice vs GrLivArea, colored by FullBath ---
bath_vals = df_clean['FullBath'].values
bath_unique = sorted(df_clean['FullBath'].unique())
bath_colors = plt.cm.viridis(np.linspace(0, 1, len(bath_unique)))
bath_cmap = {v: bath_colors[i] for i, v in enumerate(bath_unique)}
colors1 = [bath_cmap[b] for b in bath_vals]

sc1 = axes[0, 0].scatter(df_clean['GrLivArea'], df_clean['SalePrice'],
                          c=bath_vals, cmap='viridis', alpha=0.6, s=25,
                          vmin=bath_unique[0], vmax=bath_unique[-1])
cbar1 = plt.colorbar(sc1, ax=axes[0, 0])
cbar1.set_label('FullBath', fontsize=9)
cbar1.set_ticks(bath_unique)
z = np.polyfit(df_clean['GrLivArea'], df_clean['SalePrice'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_clean['GrLivArea'].min(), df_clean['GrLivArea'].max(), 200)
axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.9, linewidth=2, label='Trend line')
axes[0, 0].set_xlabel('GrLivArea (Sq Ft)')
axes[0, 0].set_ylabel('SalePrice ($)')
axes[0, 0].set_title(f'H1: SalePrice vs GrLivArea  (color = FullBath)\nr = {correlation_matrix.loc["SalePrice","GrLivArea"]:.3f}')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)

# --- H2: SalePrice vs FullBath, colored by GrLivArea ---
sc2 = axes[0, 1].scatter(df_clean['FullBath'], df_clean['SalePrice'],
                          c=df_clean['GrLivArea'], cmap='plasma', alpha=0.6, s=25)
cbar2 = plt.colorbar(sc2, ax=axes[0, 1])
cbar2.set_label('GrLivArea (sq ft)', fontsize=9)
axes[0, 1].set_xlabel('FullBath (Count)')
axes[0, 1].set_ylabel('SalePrice ($)')
axes[0, 1].set_title(f'H2: SalePrice vs FullBath  (color = GrLivArea)\nr = {correlation_matrix.loc["SalePrice","FullBath"]:.3f}')
axes[0, 1].grid(alpha=0.3)
# Jitter for clarity on discrete x-axis
jitter = np.random.uniform(-0.1, 0.1, size=len(df_clean))
axes[0, 1].scatter(df_clean['FullBath'] + jitter, df_clean['SalePrice'],
                   c=df_clean['GrLivArea'], cmap='plasma', alpha=0.4, s=15)

# --- H3: SalePrice vs YearBuilt, colored by BedroomAbvGr ---
sc3 = axes[1, 0].scatter(df_clean['YearBuilt'], df_clean['SalePrice'],
                          c=df_clean['BedroomAbvGr'], cmap='coolwarm', alpha=0.6, s=25)
cbar3 = plt.colorbar(sc3, ax=axes[1, 0])
cbar3.set_label('BedroomAbvGr', fontsize=9)
z = np.polyfit(df_clean['YearBuilt'], df_clean['SalePrice'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_clean['YearBuilt'].min(), df_clean['YearBuilt'].max(), 200)
axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.9, linewidth=2, label='Trend line')
axes[1, 0].set_xlabel('YearBuilt')
axes[1, 0].set_ylabel('SalePrice ($)')
axes[1, 0].set_title(f'H3: SalePrice vs YearBuilt  (color = BedroomAbvGr)\nr = {correlation_matrix.loc["SalePrice","YearBuilt"]:.3f}')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend(fontsize=8)

# --- H4: SalePrice vs BedroomAbvGr, colored by YearBuilt ---
sc4 = axes[1, 1].scatter(df_clean['BedroomAbvGr'], df_clean['SalePrice'],
                          c=df_clean['YearBuilt'], cmap='YlOrRd', alpha=0.6, s=25)
cbar4 = plt.colorbar(sc4, ax=axes[1, 1])
cbar4.set_label('YearBuilt', fontsize=9)
axes[1, 1].set_xlabel('BedroomAbvGr (Count)')
axes[1, 1].set_ylabel('SalePrice ($)')
axes[1, 1].set_title(f'H4: SalePrice vs BedroomAbvGr  (color = YearBuilt)\nr = {correlation_matrix.loc["SalePrice","BedroomAbvGr"]:.3f}')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_scatterplots.png', dpi=300, bbox_inches='tight')
print("\nEnhanced scatterplots saved as '03_scatterplots.png'")

# --- Section 7: Linear Regression ---

print("\n" + "="*60)
print("LINEAR REGRESSION ANALYSIS")
print("="*60)

# Features and target
feature_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
X = df_clean[feature_cols]
y = df_clean['SalePrice']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} homes")
print(f"Test set:     {len(X_test)} homes")

# --- Model 1: Simple regression (GrLivArea only) ---
print("\n--- Model 1: Simple Linear Regression (GrLivArea → SalePrice) ---")
model1 = LinearRegression()
model1.fit(X_train[['GrLivArea']], y_train)
y_pred1 = model1.predict(X_test[['GrLivArea']])
r2_1   = r2_score(y_test, y_pred1)
rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred1))
mae_1  = mean_absolute_error(y_test, y_pred1)
print(f"  Intercept : ${model1.intercept_:,.0f}")
print(f"  Slope     : ${model1.coef_[0]:,.2f} per sq ft")
print(f"  R²        : {r2_1:.4f}  ({r2_1*100:.1f}% of variance explained)")
print(f"  RMSE      : ${rmse_1:,.0f}")
print(f"  MAE       : ${mae_1:,.0f}")

# --- Model 2: Multiple linear regression (all 4 features) ---
print("\n--- Model 2: Multiple Linear Regression (all 4 features) ---")
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
r2_2   = r2_score(y_test, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred2))
mae_2  = mean_absolute_error(y_test, y_pred2)

print(f"\n  Intercept : ${model2.intercept_:,.0f}")
print(f"  Coefficients:")
for feat, coef in zip(feature_cols, model2.coef_):
    print(f"    {feat:<18}: ${coef:,.2f}")
print(f"\n  R²   : {r2_2:.4f}  ({r2_2*100:.1f}% of variance explained)")
print(f"  RMSE : ${rmse_2:,.0f}")
print(f"  MAE  : ${mae_2:,.0f}")
print(f"\n  R² improvement over simple model: +{(r2_2 - r2_1)*100:.1f} percentage points")

# --- Section 8: Regression Plots ---

print("\nGenerating regression diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Linear Regression Analysis — Housing Price Model', fontsize=16, fontweight='bold')

# 8a. Simple regression: actual vs GrLivArea with regression line
axes[0, 0].scatter(X_test['GrLivArea'], y_test, alpha=0.5, s=20, color='steelblue', label='Actual')
x_range = np.linspace(X_test['GrLivArea'].min(), X_test['GrLivArea'].max(), 200)
axes[0, 0].plot(x_range, model1.predict(x_range.reshape(-1, 1)), 'r-', linewidth=2.5, label=f'Model (R²={r2_1:.3f})')
axes[0, 0].set_xlabel('GrLivArea (Sq Ft)')
axes[0, 0].set_ylabel('SalePrice ($)')
axes[0, 0].set_title('Simple Regression:\nGrLivArea → SalePrice')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 8b. Multiple regression: predicted vs actual
max_val = max(y_test.max(), y_pred2.max())
axes[0, 1].scatter(y_test, y_pred2, alpha=0.5, s=20, color='mediumpurple')
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit')
axes[0, 1].set_xlabel('Actual SalePrice ($)')
axes[0, 1].set_ylabel('Predicted SalePrice ($)')
axes[0, 1].set_title(f'Multiple Regression:\nActual vs. Predicted (R²={r2_2:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 8c. Residual plot
residuals = y_test - y_pred2
axes[0, 2].scatter(y_pred2, residuals, alpha=0.5, s=20, color='coral')
axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('Predicted SalePrice ($)')
axes[0, 2].set_ylabel('Residual ($)')
axes[0, 2].set_title('Residual Plot\n(Multiple Regression)')
axes[0, 2].grid(alpha=0.3)

# 8d. Residual distribution
axes[1, 0].hist(residuals, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residual ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution\n(should be ≈ normal)')
axes[1, 0].grid(alpha=0.3)

# 8e. Coefficient importance (absolute value)
coef_df = pd.Series(np.abs(model2.coef_), index=feature_cols).sort_values(ascending=True)
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
axes[1, 1].barh(coef_df.index, coef_df.values, color=colors_bar, edgecolor='black')
axes[1, 1].set_xlabel('|Coefficient| (impact per unit)')
axes[1, 1].set_title('Feature Importance\n(absolute coefficient size)')
axes[1, 1].grid(alpha=0.3, axis='x')

# 8f. Model comparison bar chart
model_names = ['Simple\n(GrLivArea only)', 'Multiple\n(all 4 features)']
r2_scores   = [r2_1, r2_2]
bar_colors  = ['#90CAF9', '#1565C0']
bars = axes[1, 2].bar(model_names, r2_scores, color=bar_colors, edgecolor='black', width=0.5)
for bar, val in zip(bars, r2_scores):
    axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'R²={val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1, 2].set_ylabel('R² Score')
axes[1, 2].set_ylim(0, 1)
axes[1, 2].set_title('Model Comparison\n(R² Score)')
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('04_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\nRegression analysis saved as '04_regression_analysis.png'")

# --- Section 9: Predictions ---

print("\n" + "="*60)
print("SAMPLE PREDICTIONS (Multiple Regression Model)")
print("="*60)

sample_homes = pd.DataFrame({
    'GrLivArea':    [1000, 1500, 2000, 2500, 3000, 1200],
    'BedroomAbvGr': [2,    3,    3,    4,    4,    2   ],
    'FullBath':     [1,    1,    2,    2,    3,    1   ],
    'YearBuilt':    [1960, 1990, 2000, 2005, 2010, 1975],
})

preds = model2.predict(sample_homes)

print(f"\n{'GrLivArea':>10} {'Beds':>5} {'Baths':>6} {'YrBuilt':>8} | {'Predicted Price':>16}")
print("-" * 55)
for i, row in sample_homes.iterrows():
    print(f"{int(row['GrLivArea']):>10,} {int(row['BedroomAbvGr']):>5} {int(row['FullBath']):>6} {int(row['YearBuilt']):>8} | ${preds[i]:>14,.0f}")

# Confidence-style range using RMSE as ±1σ
print(f"\n  Note: ±1 RMSE uncertainty band = ±${rmse_2:,.0f}")

# --- Section 10: Hypothesis testing ---

print("\n" + "="*60)
print("HYPOTHESIS TESTING RESULTS")
print("="*60)

print("\nHYPOTHESIS 1: Larger homes (higher GrLivArea) sell for higher prices")
r1 = correlation_matrix.loc['SalePrice', 'GrLivArea']
print(f"  Pearson r : {r1:.3f}")
slope_1 = model1.coef_[0]
print(f"  Reg slope : ${slope_1:,.2f} per additional sq ft")
print(f"  Assessment: {'SUPPORTED' if r1 > 0.5 else 'WEAKLY SUPPORTED' if r1 > 0 else 'NOT SUPPORTED'}")

print("\nHYPOTHESIS 2: Homes with more full bathrooms sell for higher prices")
r2 = correlation_matrix.loc['SalePrice', 'FullBath']
coef_bath = model2.coef_[feature_cols.index('FullBath')]
print(f"  Pearson r : {r2:.3f}")
print(f"  Reg coef  : ${coef_bath:,.0f} per additional full bath (controlling for other vars)")
print(f"  Assessment: {'SUPPORTED' if r2 > 0.5 else 'WEAKLY SUPPORTED' if r2 > 0 else 'NOT SUPPORTED'}")

print("\nHYPOTHESIS 3: Newer homes (higher YearBuilt) sell for higher prices")
r3 = correlation_matrix.loc['SalePrice', 'YearBuilt']
coef_year = model2.coef_[feature_cols.index('YearBuilt')]
print(f"  Pearson r : {r3:.3f}")
print(f"  Reg coef  : ${coef_year:,.0f} per year newer (controlling for other vars)")
print(f"  Assessment: {'SUPPORTED' if r3 > 0.5 else 'WEAKLY SUPPORTED' if r3 > 0 else 'NOT SUPPORTED'}")

print("\nHYPOTHESIS 4: Bedrooms predict price even after controlling for square footage")
r4 = correlation_matrix.loc['SalePrice', 'BedroomAbvGr']
coef_bed = model2.coef_[feature_cols.index('BedroomAbvGr')]
print(f"  Pearson r : {r4:.3f}")
print(f"  Reg coef  : ${coef_bed:,.0f} per additional bedroom (controlling for other vars)")
print(f"  Note: Negative coef suggests bedrooms trade off with sq footage in the model.")
print(f"  Assessment: MIXED — bivariate correlation is positive but multiple regression coefficient is {'+' if coef_bed > 0 else ''}{coef_bed:,.0f}")

# --- Section 11: Summary ---

print(f"""
{"="*60}
FINAL SUMMARY
{"="*60}

DATASET CHARACTERISTICS:
  - Total records: {len(df_clean)}
  - Variables analyzed: {len(key_variables)}
  - Missing values: {df_clean.isnull().sum().sum()} (none in key variables)
  - Price range: ${df_clean['SalePrice'].min():,.0f} – ${df_clean['SalePrice'].max():,.0f}

MODEL PERFORMANCE:
  Simple Regression (GrLivArea only):
    R² = {r2_1:.4f}  |  RMSE = ${rmse_1:,.0f}  |  MAE = ${mae_1:,.0f}

  Multiple Regression (all 4 features):
    R² = {r2_2:.4f}  |  RMSE = ${rmse_2:,.0f}  |  MAE = ${mae_2:,.0f}

MODEL EQUATION:
  SalePrice ≈ ${model2.intercept_:,.0f}
    + ${model2.coef_[0]:,.2f} × GrLivArea
    + ${model2.coef_[1]:,.2f} × BedroomAbvGr
    + ${model2.coef_[2]:,.2f} × FullBath
    + ${model2.coef_[3]:,.2f} × YearBuilt

KEY FINDINGS:
  1. GrLivArea is the dominant predictor (r = {correlation_matrix.loc['SalePrice','GrLivArea']:.3f})
  2. Adding all 4 features explains {r2_2*100:.1f}% of price variance
  3. Each additional square foot adds ≈${model2.coef_[0]:,.0f} in value
  4. Each additional full bath adds ≈${model2.coef_[2]:,.0f} (controlled for size)

OUTPUT FILES:
  01_distributions.png      — Histograms of key variables
  02_correlation_matrix.png — Pearson correlation heatmap
  03_scatterplots.png       — Color-encoded scatter plots (3-variable)
  04_regression_analysis.png — Full regression diagnostic panel
""")

print("Done! Analysis complete.")
