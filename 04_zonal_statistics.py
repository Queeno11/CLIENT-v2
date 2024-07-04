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
TOTAL_CHUNKS = 4
print("Loading data...")

t0 = time.time()
### Global Loads
## ADM boundaries data (load the full dataset because I'll iterate over it all the times)
adm_id_full = xr.open_dataset(rf"/mnt/e/client_v2_data/WB_country_grid.nc")["ID"].load()

## Load all shocks data
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
                    rf"{DATA_PROC}/shocks/{var}_{year}_{chunk_number}_zonal_stats.parquet"
                )

        chunk_end = time.time()
        print(f"Total chunk Time: {chunk_end - chunk_start_time}")
