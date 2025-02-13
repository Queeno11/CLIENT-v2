if __name__ == "__main__":

    import os
    import gc
    import cupy as cp
    import xarray as xr
    import cupy_xarray  # Adds .cupy to Xarray objects
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
    DATA_OUT = rf"/home/nico/data"
    PARQUET_PATH = rf"{DATA_PROC}/shocks_by_adm"
    GPW_PATH = rf"/mnt/d/Datasets/Gridded Population of the World"

    TOTAL_CHUNKS = 16

    print("Loading data...")

    ### Global Loads
    # Population data is loaded in the loop

    # World Bank country bounds and IDs (we only get the total bounds from here)
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")

    # ADM boundaries data (load the full dataset because I'll iterate over it all the times)
    WB_adm_id_full = xr.open_dataset(rf"{DATA_PROC}/WB_country_grid.nc")["ID"]
    IPUMS_adm_id_full = xr.open_dataset(rf"{DATA_PROC}/IPUMS_country_grid.nc")["ID"]
    adm_grids = {
        "IPUMS": IPUMS_adm_id_full,#.isel({"x": slice(12000,14000), "y": slice(10000,12000)}),
        "WB": WB_adm_id_full,
    }

    ### Shocks
    # To add new data:
    #   - Create a yearly grid with a boolean variable (1 if the event happened, 0 otherwise)
    #   - Save it in the DATA_OUT folder
    #   - Add it to the shocks dictionary
    #   - Variables can be named whatever you want, but the grid has to have the following dimensions:
    #       - x: Longitude
    #       - y: Latitude
    #       - year: Year
    #   - The grid has to be ascending in x and descending in y.
    #   - It is recommended to chunk the grid in the same way as the data is stored (lat-lon)

    hurricanes = xr.open_dataset(rf"{DATA_OUT}/IBTrACS_hurricanes_yearly.nc")
    heatwaves = xr.open_dataset(rf"{DATA_OUT}/CCKP_heatwaves_yearly.nc")
    coldwaves = xr.open_dataset(rf"{DATA_OUT}/CCKP_coldwaves_yearly.nc")
    intenserain = xr.open_dataset(rf"{DATA_OUT}/CCKP_intenserain_yearly.nc")
    droughts = xr.open_dataset(rf"{DATA_OUT}/ERA5_droughts_yearly.nc").drop_duplicates(
        dim="x"
    )
    # floods = xr.open_dataset(rf"{DATA_OUT}/GFD_floods_yearly.nc").rename(
    #     {"band_data": "flooded"}
    # )

    shocks = {
        # "floods": floods,
        # "hurricanes": hurricanes,
        "drought": droughts,
        "heatwaves": heatwaves,
        "coldwaves": coldwaves,
        "intenserain": intenserain,
    }

    ### Run process
    for admname, adm_id_full in adm_grids.items():
        print("--------------------------------")
        print(f"---   {admname} ADM boundaries   ---")
        print("--------------------------------")
        adm_id_full = (
            adm_id_full.fillna(99999)  # Fill with 99999 to avoid float issues
            .astype(int)
            .load()
        )
        # Create chunks_path if it doesn't exist
        chunks_path = os.path.join(PARQUET_PATH, admname)
        os.makedirs(chunks_path, exist_ok=True)

        for shockname, shock in shocks.items():

            print(f"----- Processing {shockname}...")

            shock, needs_interp, needs_coarsen = utils.identify_needed_transformations(
                shock, adm_id_full
            )

             ## Loop over chunks
            #   Data is dividied in chunks (sections of the world) to avoid memory issues
            #   and to allow parallel processing. This loop will iterate over every chunk
            for chunk_number in tqdm(range(TOTAL_CHUNKS)):

                datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
                    chunk_number, total_chunks=TOTAL_CHUNKS, canvas=WB_data.total_bounds
                )

                # Load datasets in memory if there's enough space
                chunk_shock, is_loaded = utils.try_loading_ds(shock.sel(datafilter))
                chunk_adm_id = adm_id_full.sel(datafilter)
                if (chunk_adm_id != 99999).sum() == 0:
                    print("No data in this chunk, skipping...")
                    continue
                
                # This chunk is loaded in CPU memory, but we'll send it to GPU so it does not move in each loop iteration
                chunk_adm_id = chunk_adm_id.as_cupy()
                
                # Load GPW data as cupy arrays
                gpw = utils.load_gpw_data(GPW_PATH , datafilter)

                ## Loop over variables
                for var in tqdm(shock.data_vars, leave=False):
                    chunk_var = chunk_shock[var]
                    chunk_var = chunk_var.as_cupy()
                    
                    ## Loop over years
                    # Note: data in this NC file will query faster if chunked in the same way as the data is stored
                    #   so loading the chunks based on lat-lon will be fast. Once in memory, we can slice by year and
                    #   send to cupy faster. That's why we loop over chunks first and then years.
                    gpw_year_prev = 0
                    for year in tqdm(shock.year.values, leave=False):

                        out_path = rf"{chunks_path}/{admname}_{shockname}_{var}_{year}_{chunk_number}_zonal_stats.parquet"
                        # if os.path.exists(out_path):
                        #     continue

                        chunk_year_var = chunk_var.sel(year=year)

                        gpw_year = utils.find_gpw_closes_year(year)
                        if gpw_year != gpw_year_prev:
                            chunk_year_gpw = gpw[gpw_year]
                            gpw_year_prev = gpw_year

                        if needs_interp:
                            chunk_year_var = utils.interpolate_era5_data(
                                chunk_year_var,
                                chunk_adm_id,
                                verbose=False,
                            )
                        else:
                            chunk_year_var = chunk_year_var.data

                        ### Groupby
                        df = utils.compute_zonal_statistics(
                            chunk_year_var, chunk_adm_id.data, chunk_year_gpw
                        )
                        df.to_parquet(out_path)
                        
                df = None
                chunk_var = None    
                chunk_year_var = None
                gc.collect()

        chunk_adm_id = None
        adm_id_full = None
        shock = None
        gc.collect()
            
