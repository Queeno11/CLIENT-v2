

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
                                                
    return None
                    
def validate_hc_merge(df, gdf):
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
    
    id_cols = ["adm2", "adm1", "adm0"]

    # Pre-check duplicates on gdf
    gdf_duplicates = gdf.duplicated(subset=id_cols).sum()
    if gdf_duplicates > 0:
        raise ValueError(f"GeoDataFrame has duplicates. {gdf_duplicates} duplicates found.")

    gdf["gdf_ismerged"] = True
    df["df_ismerged"] = True

    # gdf.set_index(id_cols, inplace=True)
    # df.set_index(id_cols, inplace=True)
    
    # Grouping df by the unique combinations of year, threshold, measure, and variable
    selectors = ["s1", "s2", "s3", "s4", "s5", "outcome"]
    grouped = df.groupby(selectors)
    if grouped.ngroups == 0:
        raise ValueError("No groups found in the dataframe. Check the input data...")
    
    for selection, group in tqdm(grouped, desc="Checking groups"):
        
        # Reset index of the group
        # df_filtered = group.reset_index(drop=True)
        
        merged = gdf.merge(group, on=id_cols, how="outer", validate="1:1", indicator=True)
        # merged.loc[merged["gdf_ismerge"] & merged["df_ismerge"], "_merge"] = "both"
        # merged.loc[merged["gdf_ismerge"].isna() & merged["df_ismerge"], "_merge"] = "right_only"
        # merged.loc[merged["gdf_ismerge"] & merged["df_ismerge"].isna(), "_merge"] = "left_only"
        
        if merged[merged["_merge"] == "right_only"].shape[0] > 0:
            raise ValueError(
                f"Merge is not 1:1 for {selection}. The IDs do not match."
            )                    
            
        print("Number of polygons without data:", merged[merged["_merge"] == "left_only"].shape[0])
    return None