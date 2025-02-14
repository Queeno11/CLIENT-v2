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

    print("Loading data...")
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    ### Global Loads
    # Population data is loaded in the loop

    # World Bank country bounds and IDs (we only get the total bounds from here)
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")

    # ADM boundaries data (load the full dataset because I'll iterate over it all the times)
    WB_adm_id_full = xr.open_dataset(rf"{DATA_PROC}/WB_country_grid.nc", chunks="auto")["ID"]
    IPUMS_adm_id_full = xr.open_dataset(rf"{DATA_PROC}/IPUMS_country_grid.nc", chunks="auto")["ID"]
    adm_grids = {
        "IPUMS": IPUMS_adm_id_full,
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

    hurricanes = xr.open_dataset(rf"{DATA_OUT}/IBTrACS_hurricanes_yearly.nc", chunks="auto")
    heatwaves = xr.open_dataset(rf"{DATA_OUT}/CCKP_heatwaves_yearly.nc", chunks="auto")
    coldwaves = xr.open_dataset(rf"{DATA_OUT}/CCKP_coldwaves_yearly.nc", chunks="auto")
    intenserain = xr.open_dataset(rf"{DATA_OUT}/CCKP_intenserain_yearly.nc", chunks="auto")
    droughts = xr.open_dataset(rf"{DATA_OUT}/ERA5_droughts_yearly.nc", chunks="auto").drop_duplicates(
        dim="x"
    )
    floods = xr.open_dataset(rf"{PATH}/Data/Data_out/GFD_floods_yearly.nc", chunks="auto").rename(
        {"band_data": "flooded"}
    )

    shocks = {
        "floods": {"ds": floods, "chunks": 8**2},
        "drought": {"ds": droughts, "chunks": 4**2},
        "hurricanes": {"ds": hurricanes, "chunks": 6**2},
        "heatwaves": {"ds": heatwaves, "chunks": 4**2},
        "coldwaves": {"ds": coldwaves, "chunks": 4**2},
        "intenserain": {"ds": intenserain, "chunks": 4**2},
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

        for shockname, data in shocks.items():

            print(f"----- Processing {shockname}...")
            
            shock = data["ds"]
            TOTAL_CHUNKS = data["chunks"]
            shock, needs_interp, needs_coarsen = utils.identify_needed_transformations(
                shock, adm_id_full
            )

             ## Loop over chunks
            #   Data is dividied in chunks (sections of the world) to avoid memory issues
            #   and to allow parallel processing. This loop will iterate over every chunk
            for chunk_number in tqdm(range(TOTAL_CHUNKS)):

                datafilter = None
                chunk_shock = None
                no_data_in_chunk = False
                
                ## Loop over variables
                for var in tqdm(shock.data_vars, leave=False):
                    if no_data_in_chunk:
                        break
                    
                    ## Loop over years
                    # Note: data in this NC file will query faster if chunked in the same way as the data is stored
                    #   so loading the chunks based on lat-lon will be fast. Once in memory, we can slice by year and
                    #   send to cupy faster. That's why we loop over chunks first and then years.
                    chunk_var = None    
                    gpw_year_prev = 0
                    for year in tqdm(shock.year.values, leave=False):

                        out_path = rf"{chunks_path}/{admname}_{shockname}_{var}_{year}_{chunk_number}_zonal_stats.parquet"
                        if os.path.exists(out_path):
                            continue

                        if datafilter is None:
                            ## Load ADM dataset
                            datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
                                chunk_number, total_chunks=TOTAL_CHUNKS, canvas=WB_data.total_bounds
                            )

                            chunk_adm_id = adm_id_full.sel(datafilter)
                            
                            no_data_in_chunk = (chunk_adm_id != 99999).sum() == 0
                            if no_data_in_chunk:
                                # print("No data in this chunk, skipping...")
                                break
                            
                            chunk_adm_id = chunk_adm_id.as_cupy().load() # Load to VRAM

                            # Load GPW to VRAM
                            gpw = utils.load_gpw_data(GPW_PATH , datafilter)
                            
                        if chunk_shock is None:
                            ## Load datasets in memory if there's enough space
                            chunk_shock = shock.sel(datafilter)
                            # chunk_shock, is_loaded = utils.try_loading_ds(shock.sel(datafilter))

                        ## Load stuff to GPU (both elements ar cupy arrays - gwp is a dict of cupy arrays)
                        if chunk_var is None:
                            chunk_var = chunk_shock[var].as_cupy().load()

                        chunk_year_var = chunk_var.sel(year=year)

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
                            chunk_year_var, chunk_adm_id.data, gpw[utils.find_gpw_closes_year(year)]
                        )
                        df.to_parquet(out_path)
                        
                        chunk_year_var = None
                        df = None

                    chunk_var = None    
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()

                datafilter = None
                chunk_adm_id = None
                chunk_shock = None
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                gc.collect()

            shock = None
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            gc.collect()
            
        adm_id_full = None
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        gc.collect()

        