if __name__ == "__main__":

    import os
    import gc
    import time
    import psutil
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
    IPUMS_data = gpd.read_feather(rf"{DATA_PROC}/IPUMS_country_IDs.feather")
    gdfs = {"WB": WB_data, "IPUMS": IPUMS_data}

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
        # "hurricanes": hurricanes,
    }

    for admname, gdf in gdfs.items():
        chunks_path = os.path.join(PARQUET_PATH, admname)
        print("--------------------------------")
        print(f"---   {admname} ADM boundaries   ---")
        print("--------------------------------")

        for shockname, shock in shocks.items():

            print(f"Exporting {shockname}...")

            ## Save shock data
            # Compile all the dataframes and generate country dtas
            out_df, variables = utils.process_all_dataframes(
                gdf, chunks_path, shockname
            )

            # Export minimal version
            out_df.drop(columns=["geometry", "Unnamed: 0"]).to_stata(
                os.path.join(DATA_OUT, f"{admname}_{shockname}_by_admlast.dta"),
                write_index=False,
            )

            # # Export full version with geometry
            # out_df.to_feather(
            #     os.path.join(DATA_OUT, f"{shockname}_by_admlast.feather"),
            # )
            break
