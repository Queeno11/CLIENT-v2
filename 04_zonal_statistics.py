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

    ### Global Loads
    # Population data is loaded in the loop

    # World Bank country bounds and IDs (we only get the total bounds from here)
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")
    IPUMS_data = gpd.read_feather(rf"{DATA_PROC}/IPUMS_country_IDs.feather")
    gdfs = {
        "IPUMS": IPUMS_data,
        "WB": WB_data,
    }

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

            ## Loop over chunks
            #   Data is dividied in chunks (sections of the world) to avoid memory issues
            #   and to allow parallel processing. This loop will iterate over every chunk
            utils.process_shock(data, adm_id_full, chunks_path, admname, shockname, WB_data, GPW_PATH, recompute=False)
            
            print(f"Exporting {shockname}...")

            # Compile all the dataframes and generate country dtas
            outpath_long = os.path.join(DATA_OUT, f"{admname}_{shockname}_long.csv")
            out_df = utils.process_all_dataframes(gdfs[admname], chunks_path, shockname)
            out_df.to_csv(outpath_long)
            print(f"Se creó {outpath_long}")
            
            if admname == "IPUMS":
                ### STATA VERSION (WIDE) (only for IPUMS)
                outpath_wide = os.path.join(DATA_OUT, f"{admname}_{shockname}_wide.dta")
                out_df = utils.process_to_stata(out_df, gdfs[admname])
                out_df.to_stata(outpath_wide, write_index=False)
                print(f"Se creó {outpath_wide}")
                            
        adm_id_full = None
        gc.collect()

    print("Done! Now share the wide datasets with the team to run the STATA scripts in the Server.")
        