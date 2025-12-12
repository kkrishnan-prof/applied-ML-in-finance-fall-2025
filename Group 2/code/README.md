# Home Insurance Pricing Optimization

## Overview
This project uses machine learning to identify underpriced and overpriced zip codes
by analyzing wildfire risk, earthquake risk, building characteristics (from Zillow)
and census demographics. The model predicts optimal premium rates and highlights 
geographic areas where pricing adjustments are needed.

## Key Features
- Predicts insurance premiums per exposure using Random Forest, XGBoost and SVR models
- Creates a composite risk score from SHAP-weighted risk factors (wildfire, earthquake,
  building age, etc)
- Identifies UNDERPRICED and OVERPRICED zip codes with recommended rate changes
- Backtest on 2021 data with visualizations and pricing analysis

## Interactive App
https://californiazippremiumpredictor.onrender.com/

## Results
- **Best Model:** XGBoost (R^2 0.77 on test set)
- **Top Risk Factors:** Wildfire exposure, earthquake proximity, building age, population density
- **Pricing Insights:** 219 Underpriced and 171 overpriced ZIPs

## Project repo
https://github.com/sandrasri/CaliforniaHousingPremiums.git