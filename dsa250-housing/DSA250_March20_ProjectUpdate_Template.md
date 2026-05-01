# DSA 250 - Project Update
## Housing Price Analysis
**Delilah Hollander | March 26, 2026**

---

## What I'm Looking At

For this project I'm analyzing the Ames Housing Dataset from Kaggle. It has info on 1,460 home sales in Ames, Iowa. I want to figure out what makes a house more expensive.

My four research questions:
1. Does square footage predict sale price?
2. Do more bathrooms mean a higher price?
3. Are newer homes more expensive?
4. Does bedroom count matter once you account for size?

---

## The Data

I downloaded `train.csv` from Kaggle. It has 81 columns total, but I only used the 5 that made sense for my questions:

| Variable | Description |
|----------|-------------|
| SalePrice | What the house sold for (the thing I'm trying to predict) |
| GrLivArea | Square footage above ground |
| BedroomAbvGr | Number of bedrooms |
| FullBath | Number of full bathrooms |
| YearBuilt | Year the house was built |

I checked for null values using `.isnull().sum()` and all 1,460 records were complete for these columns, so I didn't need to do any imputation.

---

## What I Found (EDA)

SalePrice is right-skewed, meaning most homes are in the $130k-$215k range but there are some really expensive outliers (one sold for $755k).

**Correlation with SalePrice:**
- GrLivArea: r = 0.709 (strongest)
- FullBath: r = 0.561
- YearBuilt: r = 0.523
- BedroomAbvGr: r = 0.168 (weakest)

---

## Hypothesis Results

**H1 (Square footage):** Supported. r = 0.71 is a strong positive relationship.

**H2 (Bathrooms):** Supported. r = 0.56, bathrooms matter.

**H3 (Year built):** Supported. r = 0.52, newer homes sell for more.

**H4 (Bedrooms):** Needs more testing. r = 0.17 alone is weak, probably because bedrooms and square footage overlap a lot.

---

## Next Steps

- Build a multiple linear regression model
- Test for multicollinearity
- Make actual price predictions using the model
- Finish the final presentation

---

**Submitted:** March 26, 2026
