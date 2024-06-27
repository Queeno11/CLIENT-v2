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


def get_bounds_from_chunk_number(chunk_number, total_chunks=8):
    """Get the bounding box coordinates for a given chunk number.

    Data is divided into total_chunks chunks, each covering an 1/total_chunks of the globe.

    Parameters:
    -----------
    chunk_number: int
        Chunk number < total_chunks.
    total_chunks: int
        Total number of chunks to divide the globe into.

    Returns:
    --------
    tuple: Bounding box coordinates (left, bottom, right, top).
    """

    if chunk_number > total_chunks - 1:
        raise ValueError("Chunk number must be less than total_chunks.")

    # Define the bounding box coordinates for each chunk
    x_min = -180
    x_max = 180
    y_min = -90
    y_max = 90

    # Calculate the bounding box coordinates for the given chunk number
    x_step = (x_max - x_min) / total_chunks / 2
    y_step = (y_max - y_min) / total_chunks / 2

    left = x_min + chunk_number * x_step
    right = left + x_step
    bottom = y_min + chunk_number * y_step
    top = bottom + y_step

    return (left, bottom, right, top)


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

# Loop over chunks
chunk_number = 2
chunk_bounds = get_bounds_from_chunk_number(chunk_number, total_chunks=4)
chunk_droughts = droughts.sel(
    x=slice(chunk_bounds[0], chunk_bounds[2]), y=slice(chunk_bounds[3], chunk_bounds[1])
).load()
chunk_adm_id = adm_id_full.sel(
    x=slice(chunk_bounds[0], chunk_bounds[2]), y=slice(chunk_bounds[3], chunk_bounds[1])
)
# Note: data in this NC file will query faster if chunked in the same way as the data is stored
#   so loading the chunks based on lat-lon will be fast. Once in memory, we can slice by year and
#   send to cupy faster.
print("chunk_bounds:", chunk_adm_id)


# Loop over years
year = 1995
gpw_year_prev = 0
chunk_year_drought = chunk_droughts.sel(year=year)

gpw_year = find_gpw_closes_year(year)
if gpw_year != gpw_year_prev:
    chunk_year_gpw = load_gpw_data(
        gpw_year, bounds=chunk_bounds
    )  # left, bottom, right, top
# FIXME: Aca debería agregar un check de que las tres bases tengan tamaños similares

print("chunk_year_drought", chunk_year_drought)
print()
print("chunk_year_gpw:", chunk_year_gpw)
print()
print("chunk_adm_id:", chunk_adm_id)

# Loop over variables
for var in tqdm(droughts.data_vars):
    t1 = time.time()
    # Select indicator, year and region
    print(0)
    ### Interpolate
    datavar_interp = utils.intepolate_era5_data(
        chunk_year_drought[var],
        chunk_adm_id,
        verbose=False,
    )

    ### Groupby
    t0 = time.time()
    groups = cp.asarray(chunk_adm_id.values.flatten())
    t1 = time.time()
    print(t1 - t0)
    population_values = cp.asarray(chunk_year_gpw.flatten())
    t2 = time.time()
    print(t2 - t1)
    area_values = datavar_interp.flatten()
    t3 = time.time()
    print(t3 - t2)
    del datavar_interp
    # print(groups.shape, area_values.shape, population_values.shape)
    assert groups.shape == area_values.shape == population_values.shape

    aggs = utils.compute_zonal_statistics(groups, area_values, population_values)
    t4 = time.time()
    print(t4 - t3)
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
    ).to_parquet(rf"{DATA_PROC}/shocks/{var}_{year}_{chunk_number}_zonal_stats.parquet")
    print(time.time() - t4)
    t2 = time.time()
    print(f"Loop time: {t2 - t1}")

print(f"Total Time: {t2 - t0}")
