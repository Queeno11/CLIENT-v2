

def assert_correct_colnames(df):
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
    correct_colnames = ['adm0_code', 'adm1_code', 'adm2_code', 'year', 'variable', 'value', 'measure', 'threshold']
    
    if not all([col in correct_colnames for col in colnames]):
        raise ValueError(f"Column names are not correct. They should be: {correct_colnames}. They are: {colnames.tolist()}")
    
    return None


def assert_correct_shape(df, gdf):
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
    
    n_years = df.year.unique().shape[0]
    n_thresholds = df.threshold.unique().shape[0]
    n_IDs = gdf[['adm2_code','adm1_code','adm0_code']].drop_duplicates().shape[0]
    n_measures = df.measure.unique().shape[0]
    n_variables = df.variable.unique().shape[0]
    n_obs_expected = n_years * n_thresholds * n_IDs * n_measures * n_variables

    if not (n_obs == n_obs_expected) or (n_obs == 0):
        raise ValueError(f"Number of observations is not correct. Should be: {n_obs_expected}. Is: {n_obs}")
    
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
        raise ValueError(f"No idea why the admcodes are not matching...")
    
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
    
    gdf = gdf.reset_index(drop=True)

    # Pre-check duplicates on gdf
    gdf_duplicates = gdf.duplicated(subset=["adm2_code", "adm1_code", "adm0_code"]).sum()
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
            df_filtered[["adm2_code", "adm1_code", "adm0_code"]].values,
            gdf[["adm2_code", "adm1_code", "adm0_code"]].values
        )

        if not ids_match:
            raise ValueError(
                f"Merge is not 1:1 for year {year}, threshold {threshold}, measure {measure}, variable {variable}. The IDs do not match."
            )                    
            
        # Test 2: no duplicates
        duplicates = df_filtered.duplicated(subset=["adm2_code", "adm1_code", "adm0_code"]).sum() > 0

        if duplicates:
            raise ValueError(
                f"Merge is not 1:1 for year {year}, threshold {threshold}, measure {measure}, variable {variable}. There are {duplicates} duplicates in the filtered dataframe."
            )
                                                
    return None
                    
    