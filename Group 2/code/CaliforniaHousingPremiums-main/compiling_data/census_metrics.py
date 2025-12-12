#!/usr/bin/env python3
import pandas as pd

census_file = "data_files/census_acs_all_years.csv"
df = pd.read_csv(census_file)

numeric_cols = [
    "median_income", "population", "median_home_value", "total_housing_units",
    "median_year_built", "median_rooms", "population_density", "home_age_estimate", "acs_year"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['zip_code'] = df['zip_code'].astype(str)

essential_cols = ["zip_code", "acs_year"]
df = df.dropna(subset=essential_cols)
df = df.sort_values(by=["zip_code", "acs_year"]).reset_index(drop=True)

out_file = "census_data_cleaned.csv"
df.to_csv(out_file, index=False)
print(f"✅ Cleaned census data saved to {out_file}")
