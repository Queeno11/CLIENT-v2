import gc

PATH = "D:\World Bank\CLIENT v2"
DATA_RAW = rf"{PATH}\Data\Data_raw"
DATA_PROC = rf"{PATH}\Data\Data_proc"
DATA_OUT = rf"{PATH}\Data\Data_out"
GFD_PATH = r"D:\Datasets\Global Flood Database\gfd_v1_4"
GPW_PATH = r"D:\Datasets\Gridded Population of the World"

def assert_correct_colnames(df, dataset_name="climate"):
    ''' Ensures dataframe has the correct column names for the webpage.

        Raises a ValueError if the column names are not the following:
            ['adm0', 'adm1', 'adm2', 'year', 'variable', 'value', 'measure', 'threshold']

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked.
        
    Returns
    -------
    None
    '''
    
    colnames = df.columns
    dataset_name = dataset_name.lower()
    
    if dataset_name == "climate":
        correct_colnames = ['adm0_code', 'adm1_code', 'adm2_code', 'year', 'variable', 'value', 'measure', 'threshold']
    elif dataset_name == "hc_geo_data":
        correct_colnames = ['adm0', 'adm1', 'adm2', 's1', 's2', 's3', 's4', 's5', 'outcome', 'treatment_sub', 'dif']
    elif dataset_name == "hc_national_data":
        correct_colnames = ['adm0', 's1', 's2', 's3', 's4', 's5', 'outcome', 'treatment', 'time', 'value']
    else:
        raise ValueError("Dataset name not recognized. Please use 'climate', 'hc_geo_data', or 'hc_national_data'.")
    
    if not all([col in correct_colnames for col in colnames]):
        raise ValueError(f"Column names are not correct. They should be: {correct_colnames}. They are: {colnames.tolist()}")
    
    return None


def assert_correct_shape(df, gdf, dataset_name="climate"):
    ''' Ensures dataframe has the correct number of observations for the webpage.

        Raises a ValueError if the number of observations is not equal to 
            #ID * #year * #variable * #measure * #threshold.
        (i.e., we have data for every case of the cross-product of the above dimensions)
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked.
        
    Returns
    -------
    None
    '''
    n_obs = df.shape[0]
    
    if dataset_name=="climate":
        n_years = df.year.unique().shape[0]
        n_thresholds = df.threshold.unique().shape[0]
        n_IDs = gdf[['adm2_code','adm1_code','adm0_code']].drop_duplicates().shape[0]
        n_measures = df.measure.unique().shape[0]
        n_variables = df.variable.unique().shape[0]
        n_obs_expected = n_years * n_thresholds * n_IDs * n_measures * n_variables

        if not (n_obs == n_obs_expected) or (n_obs == 0):
            raise ValueError(f"Number of observations is not correct. Should be: {n_obs_expected}. Is: {n_obs}")

    elif dataset_name=="hc_geo_data":
        raise NotImplementedError("This function is not implemented for the 'hc_geo_data' dataset.")
    
    else:
        raise ValueError("Dataset name not recognized. Please use 'climate', 'hc_geo_data', or 'hc_national_data'.")

    return

def assert_correct_admcodes(df):
    ''' Ensures that the adm codes are correct after running expand_dataset().

        Raises a ValueError if the adm codes are not correct.'''

    adm2_match = df[(df["adm2_code"]!=df["adm2_code_y"]) & df["adm2_code_y"].notna()].shape[0] == 0
    adm1_match = df[(df["adm1_code"]!=df["adm1_code_y"]) & df["adm1_code_y"].notna()].shape[0] == 0
    adm0_match = df[(df["adm0_code"]!=df["adm0_code_y"]) & df["adm0_code_y"].notna()].shape[0] == 0

    if all([adm2_match, adm1_match, adm0_match]):
        df = df.drop(columns=["adm0_code_y", "adm1_code_y", "adm2_code_y"])
    else:
        raise ValueError("No idea why the admcodes are not matching...")
    
    return df

def validate_climate_dataset(df, gdf):
    ''' Ensures that the dataframe, after applying every filter, has a 1:1 merge with the gdf. 
    
        Raises a ValueError if the merge is not 1:1.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked.
    gdf : gpd.GeoDataFrame
        GeoDataFrame to be checked.
        
    Returns
    -------
    None
    '''
    from tqdm.autonotebook import tqdm
    import numpy as np
    id_cols = ["adm2_code", "adm1_code", "adm0_code"]
    gdf = gdf.reset_index(drop=True)

    # Pre-check duplicates on gdf
    gdf_duplicates = gdf.duplicated(subset=id_cols).sum()
    if gdf_duplicates > 0:
        raise ValueError(f"GeoDataFrame has duplicates. {gdf_duplicates} duplicates found.")
    
    # Grouping df by the unique combinations of year, threshold, measure, and variable
    grouped = df.groupby(["year", "threshold", "measure", "variable"])
    if grouped.ngroups == 0:
        raise ValueError("No groups found in the dataframe. Check the input data...")
    
    for (year, threshold, measure, variable), group in tqdm(grouped, desc="Checking groups"):
        
        # Reset index of the group
        df_filtered = group.reset_index(drop=True)
        
        ## Two tests try to ensure that the merge is 1:1

        # Test 1: Check if the two DataFrames have matching IDs
        ids_match = np.array_equal(
            df_filtered[id_cols].sort_values(by=id_cols).values,
            gdf[id_cols].sort_values(by=id_cols).values
        )

        if not ids_match:
            raise ValueError(
                f"Merge is not 1:1 for year {year}, threshold {threshold}, measure {measure}, variable {variable}. The IDs do not match."
            )                    
            
        # Test 2: no duplicates
        duplicates = df_filtered.duplicated(subset=id_cols).sum() > 0

        if duplicates:
            raise ValueError(
                f"Merge is not 1:1 for year {year}, threshold {threshold}, measure {measure}, variable {variable}. There are {duplicates} duplicates in the filtered dataframe."
            )
    
    gc.collect()                                

    return None
                    
def validate_hc_merge():
    """
    Ensures that the Dask DataFrame (df), after applying every filter, 
    has a 1:1 merge with the GeoDataFrame (gdf).
    
    Raises a ValueError if the merge is not 1:1.
    
    The grouping is now done using a new index, created by concatenating
    the values from columns s1 to s5 and outcome.
    
    Returns
    -------
    None
    """
    from tqdm.auto import tqdm
    import pandas as pd
    import polars as pl
    import time
    
    dtypes = {
        "adm0":         pl.Int32,
        "adm1":         pl.Int64,
        "adm2":         pl.Int64,
        "s1":           pl.String,
        "s2":           pl.String,
        "s3":           pl.String,
        "s4":           pl.String,
        "s5":           pl.String,
        "outcome":      pl.String,
        "diff":         pl.Float32,
        "treatment_sub":pl.String,
    }

    # Read the CSV as polars lazy, and the GeoDataFrame to memory
    df_lazy = pl.scan_csv(rf"{DATA_OUT}\for webpage\HC_geo_data.csv", schema_overrides=dtypes)
    gdf = pd.read_csv(rf"{DATA_OUT}\for webpage\HC_geo_map.csv")
    
    id_cols = ["adm2", "adm1", "adm0"]

    # Pre-check duplicates on gdf
    gdf_duplicates = gdf.duplicated(subset=id_cols).sum()
    if gdf_duplicates > 0:
        raise ValueError(f"GeoDataFrame has duplicates. {gdf_duplicates} duplicates found.")
      
    # Get the unique group indices. This should be a relatively small list.
    print("Computing unique groups...")
    shocks_col = ["s1"]
    t1 = time.time()
    shocks = df_lazy.select(shocks_col).unique(subset=shocks_col).collect(streaming=True).to_pandas()
    print(f"Unique groups computed in {time.time()-t1:.2f} seconds.")
    
    if shocks.empty:
        raise ValueError("No groups found in the dataframe. Check the input data...")
    else:
        print(f"Shocks: {len(shocks)}")
        
    # For each unique group, use index slicing (.loc) to retrieve the corresponding rows
    for shock in shocks.s1.to_list():
        for s2 in ["Area", "Population"]:
            print(shock)        
            group_ids = ["s3", "s4", "s5", "outcome"]

            df_shock = df_lazy\
                .filter((pl.col("s1") == shock) & (pl.col("s2") == s2))\
                .select(id_cols + group_ids)\
                .collect(streaming=True)\
                .to_pandas()
            
                    
            grouped = df_shock.fillna(0).groupby(group_ids)
            if grouped.ngroups == 0:
                raise ValueError("No groups found in the dataframe. Check the input data...")
            
            for selection, group in tqdm(grouped, desc="Validating merges..."):

                # Reset index of the group
                # df_filtered = group.reset_index(drop=True)
                try:
                    merged = gdf.merge(group, on=id_cols, how="outer", validate="1:1", indicator=True)
                except Exception as e:
                    print(e)
                    raise ValueError(f"Error in: {selection}, {group}")            
                        
                if merged[merged["_merge"] == "right_only"].shape[0] > 0:
                    raise ValueError(
                        f"Merge is not 1:1 for {selection}. The IDs do not match."
                    )                    
                    
            df_shock = None
            merged = None
            gc.collect()

    gc.collect()
    
    return None

def get_first_var_year_with_data(ds):
    ''' Returns the first year and variable with data (with some variability) in a DataSet.
    
    Parameters
    ----------
    ds : xarray.DataSet
        DataSet to be checked.
        
    Returns
    -------
    int
        First year with data.
    '''

    first_var_with_data = None
    
    for var in list(ds.data_vars):

        da = ds[var]
        first_year_with_data = get_first_year_with_data(da) 
        if first_year_with_data is not None:
            first_var_with_data = var
            break       
            
    return first_var_with_data, first_year_with_data


def get_first_year_with_data(da):
    ''' Returns the first year with data (with some variability) in a DataArray.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray to be checked.
        
    Returns
    -------
    int
        First year with data.
    '''

    first_year_with_data = None
    
    first_year = da["year"].min().values
    last_year = da["year"].max().values
    assert first_year < last_year, "First year is greater than last year."
    for year in range(first_year, last_year):
        has_data = da.sel(year=year).max().item()

        if bool(has_data) is True: # If the maximum is True
            first_year_with_data = year
            break
        
    if first_year_with_data is None:
        raise ValueError("No year with data found.")
            
    return first_year_with_data

def get_first_chunk_with_data(da, total_chunks, canvas):
    ''' Returns the first chunk with data (with some variability) in a DataArray.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray to be checked.
    total_chunks : int
        Total number of chunks.
        
    Returns
    -------
    int
        First chunk with data.
    '''
    
    import utils 
    from math import sqrt
    
    first_chunk_with_data = None
    min_chunk = int(sqrt(total_chunks))
    last_chunk = total_chunks

    for chunk_number in range(min_chunk, last_chunk):

        datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
            chunk_number, total_chunks=total_chunks, canvas=canvas
        )
        if bool(da.sel(datafilter).max().item()) is True: # If the maximum is True
            first_chunk_with_data = chunk_number
            break

    if first_chunk_with_data is None:
        raise ValueError("No chunk with data found.")

    return first_chunk_with_data

def get_file_by_shockname(shockname):
    ''' Get the filename for a given shockname.'''
    import os
    
    files = os.listdir(DATA_OUT)
    files = [f for f in files if ".nc" in f] # Filter only netcdf files
    files = [f for f in files if shockname in f] # Filter by shockname
    
    if len(files) == 0:
        raise ValueError(f"No files found for shockname {shockname}.")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found for shockname {shockname}.")
    
    return files[0]
    
def compare_xarray_with_zonal_statistics(adm="WB", chunk_number=None, shockname="coldwaves", var=None, total_chunks=16, year=None, out_name="compare_xarray_with_zonal_statistics"):
    
    assert adm in ["WB", "IPUMS"], "adm must be either 'WB' or 'IPUMS'"
    
    import utils
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    filename = get_file_by_shockname(shockname)
    WB_data = gpd.read_feather(rf"{DATA_PROC}/WB_country_IDs.feather")
    if adm=="WB":
        adm_data = WB_data
    elif adm=="IPUMS":  
        adm_data = gpd.read_feather(rf"{DATA_PROC}/IPUMS_country_IDs.feather")
    else:
        raise ValueError("adm has to be either 'WB' or 'IPUMS'")
    
    shock = xr.open_dataset(rf"{DATA_OUT}/{filename}")
    if shockname=="floods":
        # FIXME: this should be done earlier...
        shock = shock.rename(
            {"band_data": "flooded"}
        )
    
    if year is None:
        if var is None:
            var, year = get_first_var_year_with_data(shock)
        else:
            year = get_first_year_with_data(shock[var])
        
    da = shock[var].sel(year=year)
        
    if chunk_number is None:
        chunk_number = get_first_chunk_with_data(da, total_chunks, canvas=WB_data.total_bounds)
        
    print(f"Var: {var}, Year: {year}, Chunk: {chunk_number}")
    # Load zonal_statistics output
    df = pd.read_parquet(rf"D:\World Bank\CLIENT v2\Data\Data_proc\shocks_by_adm\{adm}\{adm}_{shockname}_{var}_{year}_{chunk_number}_zonal_stats.parquet")
    merged = adm_data.set_index("ID").join(df, how="inner")

    # Load shock raster data (xarray)
    datafilter, chunk_bounds = utils.get_filter_from_chunk_number(
        chunk_number, total_chunks=total_chunks, canvas=adm_data.total_bounds
    )
    da = da.sel(datafilter)

    # Plot

    merged["area_affected"] = (merged["cells_affected"] / merged["total_cells"]).fillna(0)
    merged["population_affected"] = (merged["population_affected_n"] / merged["total_population"]).fillna(0)
    ax = merged.plot(column="area_affected", figsize=(60, 20))

    da.plot(ax=ax, alpha=0.5)

    # Add the area_affected value at the centroid of each polygon
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        if row.geometry is not None and row.geometry.is_valid:
            centroid = row.geometry.centroid
            value = int(row['area_affected'] * 100)
            ax.text(
                centroid.x, centroid.y, f"{value}",
                horizontalalignment='center',
                fontsize=4,
                color='black'
            )

    plt.savefig(rf"{DATA_OUT}/{out_name}.png", dpi=300)

if __name__ == "__main__":
    import pandas as pd

    print("Testing CLIMATE DATASET:")
    # Load the data
    gdf = pd.read_csv(r"D:\World Bank\CLIENT v2\Data\Data_out\for webpage\WB_map.csv")

    for shock in ["floods", "drought", "hurricanes", "intenserain", "heatwaves", "coldwaves"]:
        
        print("Verifying", shock)
        df = pd.read_csv(rf"D:\World Bank\CLIENT v2\Data\Data_out\for webpage\WB_{shock}.csv")
        
        assert_correct_colnames(df)
        assert_correct_shape(df, gdf)
        validate_climate_dataset(df, gdf)
        
        if shock=="floods":
            total_chunks=8**2
        elif shock=="hurricanes":
            total_chunks=6**2
        else: 
            total_chunks=4**2
            
        compare_xarray_with_zonal_statistics(adm="WB", chunk_number=5, shock="coldwaves", var="fd20", total_chunks=total_chunks, year=2004, out_name="compare_xarray_with_zonal_statistics")
        
    print("Testing HC DATASET:")
    
    for shock in ["floods", "drought", "hurricanes", "intenserain", "heatwaves", "coldwaves"]:
        compare_xarray_with_zonal_statistics(adm="IPUMS", chunk_number=5, shock="coldwaves", var="fd20", total_chunks=total_chunks, year=2004, out_name="compare_xarray_with_zonal_statistics")

    print("Rest of the dataset is validated during generation due to memory constraints.")
