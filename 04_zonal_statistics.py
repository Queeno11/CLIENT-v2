import time
import cupy as cp
import numpy as np
import xarray as xr
import pandas as pd
import utils
from tqdm import tqdm

# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import seaborn as sns

PATH = "/mnt/d/World Bank/CLIENT v2"
DATA_RAW = rf"{PATH}/Data/Data_raw"
DATA_PROC = rf"{PATH}/Data/Data_proc"
DATA_OUT = rf"{PATH}/Data/Data_out"

print("Loading data...")

t0 = time.time()
### Global Loads
## ADM boundaries data (load the full dataset because I'll iterate over it all the times)
adm_id_full = xr.open_dataset(rf"/mnt/e/client_v2_data/WB_country_grid.nc")
adm_id_full = adm_id_full["ID"].load()

## Shocks data
droughts = xr.open_dataset(
    rf"/mnt/d/World Bank/CLIENT v2/Data/Data_out/ERA5_droughts_yearly.nc"
)
floods = xr.open_dataset(
    rf"/mnt/d/World Bank/CLIENT v2/Data/Data_out/GPW_floods_yearly.nc"
).rename({"band_data": "flooded"})

shocks = [floods]

### Run process
for i, shock in enumerate(shocks):

    print(f"Processing shock {i}...")

    shock, needs_interp, need_coarsen = utils.identify_needed_transformations(
        shock, adm_id_full
    )

    # Loop over chunks
    total_chunks = 16
    for chunk_number in tqdm(range(total_chunks)):
        chunk_start_time = time.time()

        datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
            chunk_number, total_chunks=total_chunks
        )
        chunk_shock = shock.sel(datafilter).load()
        chunk_adm_id = adm_id_full.sel(datafilter).load()

        if chunk_adm_id.notnull().sum() == 0:
            print("No data in this chunk, skipping...")
            continue

        # Note: data in this NC file will query faster if chunked in the same way as the data is stored
        #   so loading the chunks based on lat-lon will be fast. Once in memory, we can slice by year and
        #   send to cupy faster.
        for year in tqdm(shock.year.values, leave=False):
            # Loop over years
            gpw_year_prev = 0
            chunk_year_shock = chunk_shock.sel(year=year)

            gpw_year = utils.find_gpw_closes_year(year)
            if gpw_year != gpw_year_prev:
                chunk_year_gpw = utils.load_gpw_data(
                    gpw_year, bounds=chunk_bounds
                )  # left, bottom, right, top

            start = time.time()
            # Loop over variables
            for var in tqdm(shock.data_vars, leave=False):

                # Select indicator, year and region
                ### Interpolate
                datavar = chunk_year_shock[var]
                if needs_interp:
                    datavar = utils.intepolate_era5_data(
                        datavar,
                        chunk_adm_id,
                        verbose=False,
                    )
                else:
                    datavar = datavar.values

                ### Groupby
                groups = cp.asarray(chunk_adm_id.values.flatten())
                population_values = cp.asarray(chunk_year_gpw.flatten())
                area_values = cp.asarray(datavar.flatten())
                del datavar
                assert (
                    groups.shape == area_values.shape == population_values.shape
                ), f"There's something wrong with the shapes of the data, they all must match: {groups.shape}, {area_values.shape}, {population_values.shape}"

                aggs = utils.compute_zonal_statistics(
                    groups, area_values, population_values
                )
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
                ).to_parquet(
                    rf"{DATA_PROC}/shocks/{var}_{year}_{chunk_number}_zonal_stats.parquet"
                )

        chunk_end = time.time()
        print(f"Total chunk Time: {chunk_end - chunk_start_time}")
