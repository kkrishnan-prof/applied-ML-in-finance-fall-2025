#!/usr/bin/env python3

import geopandas as gpd
import fiona
import pandas as pd
import os
import re

ZCTA_PATH = "data_files/tl_2024_us_zcta520"

# ---------------------------------------------
# AUTO-DETECT DATE COLUMNS
# ---------------------------------------------
def detect_date_columns(df):
    start_candidates = [
        c for c in df.columns if re.search(r"start|ignit|begin", c, re.IGNORECASE)
    ]
    end_candidates = [
        c for c in df.columns if re.search(r"end|out|control|finish|completed", c, re.IGNORECASE)
    ]

    start_col = start_candidates[0] if start_candidates else None
    end_col   = end_candidates[0] if end_candidates else None

    print("\n🟦 Detected date fields:")
    print("  Start date:", start_col)
    print("  End date:  ", end_col)

    return start_col, end_col


# ---------------------------------------------
# AUTO-DETECT ZIP CODE COLUMN
# ---------------------------------------------
def detect_zip_column(zcta_df):
    print("\n🟩 Detecting ZIP code column…")
    print("ZCTA Columns:", list(zcta_df.columns))

    zip_candidates = []

    for col in zcta_df.columns:
        try:
            sample = str(zcta_df[col].dropna().iloc[0])
        except:
            continue

        if re.fullmatch(r"\d{5}", sample):
            zip_candidates.append(col)

        if sample.replace(".", "", 1).isdigit():
            if re.fullmatch(r"\d{5}", str(int(float(sample)))):
                zip_candidates.append(col)

    if not zip_candidates:
        raise ValueError("❌ Could not detect a ZIP column in ZCTA shapefile.")

    print("  ZIP column detected:", zip_candidates[0])
    return zip_candidates[0]


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------
def convert_fire_gdb_to_zip_events(gdb_path):
    if not os.path.exists(gdb_path):
        raise FileNotFoundError(f"GDB not found: {gdb_path}")

    layers = fiona.listlayers(gdb_path)
    print("Layers found:", layers)

    layer = layers[0]
    print(f"Using layer: {layer}")

    gdf = gpd.read_file(gdb_path, layer=layer).to_crs(4326)

    # Detect date fields
    start_col, end_col = detect_date_columns(gdf)

    # Convert date columns
    for col in [start_col, end_col]:
        if col:
            gdf[col] = pd.to_datetime(gdf[col], errors="coerce").dt.tz_localize(None)

    # Duration
    if start_col and end_col:
        gdf["duration_days"] = (gdf[end_col] - gdf[start_col]).dt.days
    else:
        gdf["duration_days"] = None

    # Standard acreage field names
    gdf.rename(columns={
        "gis_acres": "acres",
        "treated_ac": "treated_acres"
    }, inplace=True)

    # Load ZIP shapes
    print("Loading ZCTA polygons…")
    zcta = gpd.read_file(ZCTA_PATH)

    # Detect ZIP column
    zip_col = detect_zip_column(zcta)

    zcta["zip_code"] = zcta[zip_col].astype(str)
    zcta = zcta[["zip_code", "geometry"]]

    # Spatial join
    joined = gpd.sjoin(
        gdf,
        zcta,
        how="left",
        predicate="intersects"
    )

    # Columns for final dataset
    keep_cols = [
        "zip_code", start_col, end_col, "duration_days",
        "acres", "treated_acres", "unit_id", "agency"
    ]
    keep_cols = [c for c in keep_cols if c in joined.columns]

    final = joined[keep_cols].copy()

    # Optional severity
    severity_cols = [c for c in joined.columns if "severity" in c.lower()]
    if severity_cols:
        final["severity"] = joined[severity_cols[0]]

    out = "wildfire_zip_events.csv"
    final.to_csv(out, index=False)

    print(f"\n🔥 Exported ZIP-coded wildfire events → {out}")

    return out, start_col, end_col


# ---------------------------------------------
# POST-CLEANING: DROP ANY WITH MISSING ZIP OR FIELDS
# ---------------------------------------------
if __name__ == "__main__":
    out_file, start_col, end_col = convert_fire_gdb_to_zip_events("data_files/fire24_1.gdb")

    df = pd.read_csv(out_file)

    # Required columns (but only keep ones that actually exist)
    required_columns = [
        "zip_code",
        start_col,
        end_col,
        "duration_days",
        "treated_acres"
    ]

    # Filter to columns that exist in the DataFrame
    required_columns = [c for c in required_columns if c in df.columns]

    # Add severity *only if present*
    if "severity" in df.columns:
        required_columns.append("severity")

    print("\n🧹 Dropping rows based on existing required fields:")
    print(required_columns)

    # Drop rows missing ANY required field
    df_clean = df.dropna(subset=required_columns)

    # Save final
    df_clean.to_csv("wildfire_by_zip.cleaned.csv", index=False)

    print("\n✅ Clean dataset saved → wildfire_by_zip.cleaned.csv")
