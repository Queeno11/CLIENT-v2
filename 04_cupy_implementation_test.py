import time
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import cupy_xarray

# PATH = "Z:\Laboral\World Bank\CLIENT v2"
# DATA_RAW = rf"{PATH}\Data\Data_raw"
# DATA_PROC = rf"{PATH}\Data\Data_proc"
# DATA_OUT = rf"{PATH}\Data\Data_out"

spei1 = xr.open_dataset(rf"/mnt/e/test.nc")
adm2 = xr.open_dataset(rf"/mnt/e/test_adm2.nc")
# WB_country_grid = xr.open_dataset(rf"{DATA_PROC}\WB_country_grid.nc")

# # Selection
# era5 = era5.sel(x=slice(-100, -60), y=slice(-5, -45))
# WB_country_grid = WB_country_grid.sel(x=slice(-100, -60), y=slice(-5, -45))
# spei1 = era5["SPEI-1"].load()
# adm2 = WB_country_grid["ADM2_CODE"].load()
# spei1 = spei1.interp_like(adm2, method="nearest")

# Profile the time it takes to compute the mean
t0 = time.time()
aggs = spei1.groupby(adm2).mean().to_dataframe()
t1 = time.time()
print(f"Time CPU: {t1 - t0}")


t0 = time.time()
spei1 = spei1.as_cupy()
adm2 = adm2.as_cupy()
aggs = spei1.groupby(adm2).mean().to_dataframe()
t1 = time.time()
print(f"Time GPU: {t1 - t0}")

aggs.to_csv(rf"/mnt/e/test2.csv")
