import time
import cupy as cp
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import rioxarray
import utils
from tqdm import tqdm

# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import seaborn as sns

PATH = "/mnt/z/Laboral/World Bank/CLIENT v2"
DATA_RAW = rf"{PATH}/Data/Data_raw"
DATA_PROC = rf"{PATH}/Data/Data_proc"
DATA_OUT = rf"{PATH}/Data/Data_out"


def find_gpw_closes_year(year):
    """Find the closest year in the GPW dataset to the given year.

    GPW data comes from 2000, 2005, 2010, 2015 and 2020.

    Parameters:
    -----------
    year: int
        Year to find the closest GPW data.

    Returns:
    --------
    int: Closest year in the GPW dataset.
    """

    years = [2000, 2005, 2010, 2015, 2020]
    return min(years, key=lambda x: abs(x - year))


def load_gpw_data(year, bounds=None):
    """Load the GPW data for given year as a CuPy array.

    If bounds are provided, the data is clipped to the bounding box.

    Parameters:
    -----------
    year: int
        Year of the GPW data to load.
    bounds: tuple
        Bounding box coordinates (left, bottom, right, top) to load the data.
    """
    if year not in [2000, 2005, 2010, 2015, 2020]:
        raise ValueError("Year must be one of 2000, 2005, 2010, 2015 or 2020.")

    with rasterio.open(
        rf"/mnt/e/client_v2_data/gpw_v4_population_count_rev11_{year}_30_sec_proc.tif",
    ) as src:
        window = None
        if bounds is not None:
            # Read the data into a NumPy array
            window = rasterio.windows.from_bounds(
                left=bounds[0],
                bottom=bounds[1],
                right=bounds[2],
                top=bounds[3],
                transform=src.transform,
            )
        data = src.read(1, window=window)  # Assuming single band, read the first band
        # Convert the NumPy array to a CuPy array
        gpw = cp.asarray(data, dtype=np.uint32)

    return gpw


print("Loading data...")

t0 = time.time()
### Global Loads
## ADM boundaries data (load the full dataset because I'll iterate over it all the times)
adm_id_full = xr.open_dataset(rf"/mnt/e/client_v2_data/WB_country_grid.nc")
adm_id_full = adm_id_full["ID"].load()

## Droughts data
droughts = xr.open_dataset(rf"/mnt/e/client_v2_data/ERA5_droughts_1970-2021.nc")


### Run process
chunk = 0
gpw_year_prev = 0
year = 1995
droughts = droughts.sel(year=year)
gpw_year = find_gpw_closes_year(year)
if gpw_year != gpw_year_prev:
    gpw = load_gpw_data(gpw_year, bounds=(90, -90, 180, 90))  # left, bottom, right, top
# FIXME: Aca debería agregar un check de que las tres bases tengan tamaños similares

for var in tqdm(droughts.data_vars):
    t1 = time.time()
    # Select indicator, year and region
    data_var = (droughts[var].sel(x=slice(90, 180), y=slice(None, None))).load()

    ### Interpolate
    datavar_interp = utils.intepolate_era5_data(
        data_var, adm_id_full.sel(x=slice(90, 180)), verbose=False
    )

    ### Groupby
    groups = cp.asarray(adm_id_full.sel(x=slice(90, 180)).values.flatten())
    population_values = cp.asarray(gpw.flatten())
    area_values = datavar_interp.flatten()
    del datavar_interp
    # print(groups.shape, area_values.shape, population_values.shape)
    assert groups.shape == area_values.shape == population_values.shape

    aggs = utils.compute_zonal_statistics(groups, area_values, population_values)
    pd.DataFrame.from_dict(
        aggs,
        orient="index",
        columns=[
            "area_affected",
            "cells_affected",
            "total_cells",
            "population_affected",
            "population_affected_n",
            "total_population",
        ],
    ).to_parquet(rf"{DATA_PROC}/shocks/{var}_{year}_{chunk}_zonal_stats.parquet")

    t2 = time.time()
    print(f"Loop time: {t2 - t1}")

print(f"Total Time: {t2 - t0}")
