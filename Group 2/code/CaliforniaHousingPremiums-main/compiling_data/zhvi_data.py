#!/usr/bin/env python3
import pandas as pd
import os

# ----------------------------
# Paths
# ----------------------------
ZHVI_PATH = "data_files/ZHVI.csv"
ZCTA_COUNTY_PATH = "data_files/zcta_county_rel_10.txt"
OUTPUT_ZHVI = "output/ca_zhvi_yearly.csv"

# ----------------------------
# 1. Load ZIP → County mapping
# ----------------------------
def load_zcta_mapping():
    zcta = pd.read_csv(ZCTA_COUNTY_PATH, dtype=str)
    zcta_ca = zcta[zcta['STATE'] == '06'][['ZCTA5', 'COUNTY']].copy()
    zcta_ca.rename(columns={'ZCTA5': 'zip_code'}, inplace=True)
    zcta_ca['COUNTY'] = zcta_ca['COUNTY'].str.zfill(3)
    return zcta_ca

# ----------------------------
# 2. Process ZHVI
# ----------------------------
def process_zhvi(file_path, zcta_ca):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ZHVI file missing: {file_path}")

    df = pd.read_csv(file_path, dtype=str)

    # Clean County column
    df['CountyName'] = df['CountyName'].str.strip().str.replace(' County','', regex=False)

    # Map County Name → FIPS
    CA_COUNTY_FIPS_TO_NAME = {
        "001": "Alameda","003": "Alpine","005": "Amador","007": "Butte",
        "009": "Calaveras","011": "Colusa","012": "Contra Costa","013": "Del Norte",
        "015": "El Dorado","017": "Fresno","019": "Glenn","021": "Humboldt",
        "023": "Imperial","025": "Inyo","027": "Kern","029": "Kings",
        "031": "Lake","033": "Lassen","035": "Los Angeles","037": "Madera",
        "039": "Marin","041": "Mariposa","043": "Mendocino","045": "Merced",
        "047": "Modoc","049": "Mono","051": "Monterey","053": "Napa",
        "055": "Nevada","057": "Orange","059": "Placer","061": "Plumas",
        "063": "Riverside","065": "Sacramento","067": "San Benito","069": "San Bernardino",
        "071": "San Diego","073": "San Francisco","075": "San Joaquin","077": "San Luis Obispo",
        "079": "San Mateo","081": "Santa Barbara","083": "Santa Clara","085": "Santa Cruz",
        "087": "Shasta","089": "Sierra","091": "Siskiyou","093": "Solano",
        "095": "Sonoma","097": "Stanislaus","099": "Sutter","101": "Tehama",
        "103": "Trinity","105": "Tulare","107": "Tuolumne","109": "Ventura",
        "111": "Yolo","113": "Yuba"
    }
    NAME_TO_FIPS = {v: k for k, v in CA_COUNTY_FIPS_TO_NAME.items()}

    df['COUNTY'] = df['CountyName'].map(NAME_TO_FIPS)
    df = df.dropna(subset=['COUNTY'])

    # Merge ZIP codes
    df_zip = df.merge(zcta_ca, on='COUNTY', how='left')
    df_zip = df_zip.dropna(subset=['zip_code'])

    # Drop unnecessary columns
    drop_cols = ['RegionID','SizeRank','RegionName','RegionType','StateName','State',
                 'City','Metro','CountyName','COUNTY']
    df_zip = df_zip.drop(columns=[c for c in drop_cols if c in df_zip.columns])

    # Melt to long format (date × zip)
    df_long = df_zip.melt(id_vars=['zip_code'], var_name='date', value_name='ZHVI_value')
    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    df_long['year'] = df_long['date'].dt.year
    df_long['ZHVI_value'] = pd.to_numeric(df_long['ZHVI_value'], errors='coerce')

    # Aggregate by zip_code × year
    df_yearly = df_long.groupby(['zip_code', 'year'], as_index=False)['ZHVI_value'].median()
    
    # Sort by ZIP and year
    df_yearly = df_yearly.sort_values(['zip_code', 'year']).reset_index(drop=True)
    
    return df_yearly

# ----------------------------
# 3. Run
# ----------------------------
if __name__ == "__main__":
    zcta_ca = load_zcta_mapping()
    zhvi_clean = process_zhvi(ZHVI_PATH, zcta_ca)
    os.makedirs("output", exist_ok=True)
    zhvi_clean.to_csv(OUTPUT_ZHVI, index=False)
    print(f"✅ Saved cleaned yearly ZHVI data → {OUTPUT_ZHVI}")