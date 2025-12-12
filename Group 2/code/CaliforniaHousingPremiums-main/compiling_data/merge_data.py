import pandas as pd

prem_df = pd.read_csv('output/kaggle_premiums.csv')
zhvi_df = pd.read_csv('output/ca_zhvi_yearly.csv')
census_df = pd.read_csv('output/census_data_cleaned.csv')
wildfire_df = pd.read_csv('output/wildfire_by_zip_cleaned.csv')
fault_df = pd.read_csv('output/zip_faults_cleaned.csv')

#Standardize column names for merge
prem_df.rename(columns={'ZIP Code': 'zip_code', 'Year': 'year'}, inplace=True)
zhvi_df.rename(columns={'zip_code': 'zip_code', 'year': 'year'}, inplace=True)
census_df.rename(columns={'zip_code': 'zip_code', 'acs_year': 'year'}, inplace=True)

# ZIP codes
prem_df['zip_code'] = prem_df['zip_code'].astype(int)
zhvi_df['zip_code'] = zhvi_df['zip_code'].astype(int)
census_df['zip_code'] = census_df['zip_code'].astype(int)
fault_df['zip_code'] = fault_df['zip_code'].astype(int)

# Years
prem_df['year'] = prem_df['year'].astype(int)
zhvi_df['year'] = zhvi_df['year'].astype(int)

#Aggregate values
wildfire_df['START_DATE'] = pd.to_datetime(wildfire_df['START_DATE'])
wildfire_df['year'] = wildfire_df['START_DATE'].dt.year
wildfire_agg = wildfire_df.groupby(['zip_code', 'year']).agg(
    total_wildfire_days=('duration_days', 'sum'),
    wildfire_events=('duration_days', 'count')
).reset_index()

wildfire_agg['zip_code'] = wildfire_agg['zip_code'].astype(int)
wildfire_agg['year'] = wildfire_agg['year'].astype(int)

#Merging datasets
df = prem_df.merge(zhvi_df, on=['zip_code', 'year'], how='left')
if 'year' in census_df.columns:
    census_df = census_df.drop(columns=['year'])
df = df.merge(census_df, on='zip_code', how='left')
if 'year' not in df.columns:
    raise ValueError("Column 'year' missing in merged DataFrame before wildfire merge")
df['year'] = df['year'].astype(int)
df = df.merge(wildfire_agg, on=['zip_code', 'year'], how='left')
df = df.merge(fault_df, on='zip_code', how='left')

#Deal with missing data
df.fillna(0, inplace=True)

#Save full datafile
df.to_csv('output/merged_zip_level_data.csv', index=False)

