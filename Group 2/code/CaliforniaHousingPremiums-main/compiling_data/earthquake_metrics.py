#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
import os

fault_file = "data_files/qfault.gdb"
fault_layer = "Qfaults_2020"
zip_centroids = "data_files/zip_centroids.csv"
output = "output/zip_faults_cleaned.csv"

if __name__ == "__main__":
    faults = gpd.read_file(fault_file, layer=fault_layer)
    faults = faults.to_crs(4326)
    df = pd.read_csv(zip_centroids)
    zips = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    zip_gdf = zips.to_crs(3857)
    fault_gdf = faults.to_crs(3857)
    fault_union = fault_gdf.unary_union
    zip_gdf["meters_to_fault"] = zip_gdf.geometry.distance(fault_union)
    zip_gdf["km_to_fault"] = zip_gdf["meters_to_fault"] / 1000
    zip_gdf["fault_nearby"] = zip_gdf["km_to_fault"] <= 10
    final = zip_gdf[["zip", "lat", "lon", "km_to_fault", "fault_nearby"]].copy()
    final.rename(columns={"zip": "zip_code"}, inplace=True)
    final = final.dropna(subset=["zip_code", "lat", "lon", "km_to_fault"])
    final.to_csv(output, index=False)