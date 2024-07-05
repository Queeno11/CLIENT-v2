import os
import time
import cupy as cp
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
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
TOTAL_CHUNKS = 16
PARQUET_PATH = rf"{DATA_PROC}/shocks"
print("Loading data...")

## Global Loads
# ADM boundaries data (load the full dataset because I'll iterate over it all the times)
adm_id_full = xr.open_dataset(rf"{DATA_PROC}/WB_country_grid.nc")["ID"].load()

# Shock
droughts = xr.open_dataset(rf"{DATA_OUT}/ERA5_droughts_yearly.nc")
floods = xr.open_dataset(rf"{DATA_OUT}/GPW_floods_yearly.nc").rename(
    {"band_data": "flooded"}
)

shocks = {
    "drought": droughts,
    "floods": floods,
}

### Run process
for shockname, shock in shocks.items():

    print(f"Processing {shockname}...")

    shock, needs_interp, need_coarsen = utils.identify_needed_transformations(
        shock, adm_id_full
    )

    ## Loop over chunks
    #   Data is dividied in chunks (sections of the world) to avoid memory issues
    #   and to allow parallel processing. This loop will iterate over every chunk
    for chunk_number in tqdm(range(TOTAL_CHUNKS)):
        chunk_start_time = time.time()

        datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
            chunk_number, total_chunks=TOTAL_CHUNKS
        )
        chunk_shock = shock.sel(datafilter).load()
        chunk_adm_id = adm_id_full.sel(datafilter).load()

        if chunk_adm_id.notnull().sum() == 0:
            print("No data in this chunk, skipping...")
            continue

        ## Loop over years
        # Note: data in this NC file will query faster if chunked in the same way as the data is stored
        #   so loading the chunks based on lat-lon will be fast. Once in memory, we can slice by year and
        #   send to cupy faster. That's why we loop over chunks first and then years.
        for year in tqdm(shock.year.values, leave=False):
            gpw_year_prev = 0
            chunk_year_shock = chunk_shock.sel(year=year)

            gpw_year = utils.find_gpw_closes_year(year)
            if gpw_year != gpw_year_prev:
                chunk_year_gpw = utils.load_gpw_data(
                    gpw_year, bounds=chunk_bounds
                )  # left, bottom, right, top

            ## Loop over variables
            for var in tqdm(shock.data_vars, leave=False):

                datavar = chunk_year_shock[var]
                if needs_interp:
                    datavar = utils.intepolate_era5_data(
                        datavar,
                        chunk_adm_id,
                        verbose=False,
                    )
                else:
                    # TODO: Coarsen
                    datavar = datavar.values

                ### Groupby
                df = utils.compute_zonal_statistics(
                    datavar, chunk_adm_id, chunk_year_gpw
                )
                df.to_parquet(
                    rf"{PARQUET_PATH}/{shockname}_{var}_{year}_{chunk_number}_zonal_stats.parquet"
                )

    ## Save shock data
    # Compile all the dataframes and generate country dtas
    print("Compiling data...")
    gdf = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")
    out_df, variables = utils.process_all_dataframes(gdf, PARQUET_PATH, shockname)

    # Export minimal version
    out_df[["adm0", "adm_lst", "year", "ID"] + variables].to_stata(
        os.path.join(DATA_OUT, f"{shockname}_by_admlast.dta")
    )
    # # Export full version with geometry
    # out_df.to_feather(
    #     os.path.join(DATA_OUT, f"{shockname}_by_admlast.feather"),
    # )
