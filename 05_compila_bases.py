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

    print("Loading data...")

    ## Global Loads
    # World Bank country bounds and IDs
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")
    IPUMS_data = gpd.read_feather(rf"{DATA_PROC}/IPUMS_country_IDs.feather")
    gdfs = {
        "IPUMS": IPUMS_data,
        "WB": WB_data,
    }

    # Shock
    shocks = [
        # "floods",
        # "hurricanes",
        # "drought",
        "heatwaves",
        "coldwaves",
        "intenserain",
    ]

    for admname, gdf in gdfs.items():
        chunks_path = os.path.join(PARQUET_PATH, admname)
        print("--------------------------------")
        print(f"---   {admname} ADM boundaries   ---")
        print("--------------------------------")

        for shockname in shocks:

            print(f"Exporting {shockname}...")

            ## Save shock data

            ### CSV VERSION (LONG)
            outpath = os.path.join(DATA_OUT, f"{admname}_{shockname}_long.csv")
            # if os.path.exists(outpath):
            #     out_df = pd.read_csv(outpath)
            #     print(f"File {shockname} already exists, skipping...")
            # else:

            # Compile all the dataframes and generate country dtas
            out_df = utils.process_all_dataframes(gdf, chunks_path, shockname)
            assert out_df.area_affected.max() <= 1, "Area affected > 1"
            assert out_df.population_affected.max() <= 1, "Pop affected > 1"
            out_df.to_csv(outpath)
            print(f"Se creó {outpath}")

            if admname != "IPUMS":
                continue

            ### STATA VERSION (WIDE) (only for IPUMS)
            outpath = os.path.join(DATA_OUT, f"{admname}_{shockname}_wide.dta")
            # if os.path.exists(outpath):
            #     print(f"File {shockname} already exists, skipping...")
            #     continue
            # else:
            out_df = utils.process_to_stata(out_df, gdf, chunks_path, shockname)
            assert (
                out_df.duplicated(
                    subset=["cntry_code", "geolevel1", "geolevel2", "year"]
                ).sum()
                == 0
            ), "Duplicated rows"
            # Export minimal version
            out_df.drop(columns=["unnamed: 0", "geometry"]).to_stata(
                outpath,
                write_index=False,
            )
