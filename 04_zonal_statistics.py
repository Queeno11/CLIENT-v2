if __name__ == "__main__":

    import os
    import gc
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
    PARQUET_PATH = rf"{DATA_PROC}/shocks_by_adm"
    GPW_PATH = rf"/mnt/d/Datasets/Gridded Population of the World"

    TOTAL_CHUNKS = 16

    print("Loading data...")

    ## Global Loads
    # Population data is loaded in the loop

    # World Bank country bounds and IDs (we only get the total bounds from here)
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")

    # ADM boundaries data (load the full dataset because I'll iterate over it all the times)
    WB_adm_id_full = (
        xr.open_dataset(rf"{DATA_PROC}/WB_country_grid.nc")["ID"]
        .fillna(99999)  # Fill with 99999 to avoid float issues
        .astype(int)
    )
    IPUMS_adm_id_full = (
        xr.open_dataset(rf"{DATA_PROC}/WB_country_grid.nc")["ID"]
        .fillna(99999)  # Fill with 99999 to avoid float issues
        .astype(int)
    )
    adm_grids = {"WB": WB_adm_id_full, "IPUMS": IPUMS_adm_id_full}

    # Shock
    droughts = xr.open_dataset(rf"{DATA_OUT}/ERA5_droughts_yearly.nc").drop_duplicates(
        dim="x"
    )
    floods = xr.open_dataset(rf"{DATA_OUT}/GFD_floods_yearly.nc").rename(
        {"band_data": "flooded"}
    )
    hurricanes = xr.open_dataset(rf"{DATA_OUT}/IBTrACS_hurricanes_yearly.nc")

    shocks = {
        "drought": droughts,
        "floods": floods,
        "hurricanes": hurricanes,
    }

    ### Run process
    for admname, adm_id_full in adm_grids.items():
        adm_id_full = adm_id_full.load()
        # Create chunks_path if it doesn't exist
        chunks_path = os.path.join(PARQUET_PATH, admname)
        os.makedirs(chunks_path, exist_ok=True)

        for shockname, shock in shocks.items():

            print("################################")
            print(f"Processing {shockname}...")
            print("################################")

            shock, needs_interp, needs_coarsen = utils.identify_needed_transformations(
                shock, adm_id_full
            )

            ## Loop over chunks
            #   Data is dividied in chunks (sections of the world) to avoid memory issues
            #   and to allow parallel processing. This loop will iterate over every chunk
            for chunk_number in tqdm(range(12, TOTAL_CHUNKS)):
                chunk_start_time = time.time()

                datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
                    chunk_number, total_chunks=TOTAL_CHUNKS, canvas=WB_data.total_bounds
                )
                chunk_shock = shock.sel(datafilter).load()
                chunk_adm_id = adm_id_full.sel(datafilter)
                if (chunk_adm_id != 99999).sum() == 0:
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
                        # Load Population data as cupy array
                        chunk_year_gpw = xr.open_dataarray(
                            rf"{GPW_PATH}/gpw_v4_population_count_rev11_{gpw_year}_30_sec.tif"
                        )
                        chunk_year_gpw = chunk_year_gpw.sel(band=1).sel(datafilter)
                        chunk_year_gpw = cp.asarray(chunk_year_gpw.values)

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
                            datavar = datavar.values

                        ### Groupby
                        df = utils.compute_zonal_statistics(
                            datavar, chunk_adm_id, chunk_year_gpw
                        )
                        df.to_parquet(
                            rf"{chunks_path}/{admname}_{shockname}_{var}_{year}_{chunk_number}_zonal_stats.parquet"
                        )

        # Cleanup variables
        adm_id_full = None
        chunk_shock = None
        chunk_adm_id = None
        chunk_year_shock = None
        chunk_year_gpw = None
        datavar = None
        gc.collect()

        for shockname, shock in shocks.items():

            print(f"Exporting {shockname}...")

            ## Save shock data
            # Compile all the dataframes and generate country dtas
            out_df, variables = utils.process_all_dataframes(
                WB_data, chunks_path, shockname
            )

            # Export minimal version
            out_df[
                ["adm0_code", "adm1_code", "adm2_code", "year", "ID"] + variables
            ].to_stata(os.path.join(DATA_OUT, f"{admname}_{shockname}_by_admlast.dta"))
            # # Export full version with geometry
            # out_df.to_feather(
            #     os.path.join(DATA_OUT, f"{shockname}_by_admlast.feather"),
            # )
            break
