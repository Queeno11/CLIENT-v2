import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import hashlib
import matplotlib.pyplot as plt
import warnings

import dask
import xarray as xr
import xrspatial
from dask.diagnostics import ProgressBar
from geocube.api.core import make_geocube

import matplotlib.pyplot as plt
import seaborn as sns

PATH = "D:\World Bank\CLIENT v2"
DATA_RAW = rf"{PATH}\Data\Data_raw"
DATA_PROC = rf"{PATH}\Data\Data_proc"
DATA_OUT = rf"{PATH}\Data\Data_out"
GPW_PATH = rf"D:\Datasets\Gridded Population of the World"

# floods = pd.read_csv(rf"{DATA_RAW}\Floods\GloFAS_floods.csv")

def load_population_data(bounds=None, generate=False):
    print("Processing Population data...")

    # Select all files in GPW folder
    files = os.listdir(GPW_PATH)
    files = [f for f in files if f.endswith(".tif")]
    
    # Compile into a single dataset
    dss = []
    for f in tqdm(files):
        
        ds = xr.open_dataset(os.path.join(GPW_PATH, f), chunks={"x": 1000, "y": 1000})
        ds["band_data"] = ds["band_data"].astype(np.uint32)
        if bounds is not None:
            ds = ds.sel(
                x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1])
            )
        if generate:
            with ProgressBar():
                ds.sel(band=1).drop_vars("band").band_data.rio.to_raster(rf"{DATA_PROC}\{f.replace('.tif','_proc.tif')}")
                print(f"Saved {f.replace('.tif','_proc.tif')}")
        
        ds["year"] = int(f.split("_")[5])
        ds = ds.set_coords('year')
        dss += [ds]
        
    population = xr.concat(dss, dim="year")    
    
    # Filter if bounds are provided
    if bounds is not None:
        population = population.sel(
            x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1])
        )
        
    # Clean band dimension
    population = population.sel(band=1).drop_vars(["band"])
    
    print("Done!")
    return population

def load_precipitation_data():
    era5 = xr.open_dataset(
        rf"{DATA_OUT}\ERA5_monthly_1970-2021_SPI-SPEI.nc",
        chunks={"latitude": 100, "longitude": 100},
    )
    era5 = era5.rename({"latitude": "y", "longitude": "x"})
    return

def load_IPUMS_country_data(wb_map):
    from osgeo import gdal
    import geopandas as gpd
    from IPython.display import display

    gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

    countries_aggregate_geolevel2 = [
        "ls", # Lesotho
        "us", # United States
        "it", # Italia
        "rw", # Rwanda
        "lr", # Liberia
        "la", # Laos
    ]
    
    print("Loading IPUMS country data...") 
    gdf = open_IPUMS(force_geolev1=countries_aggregate_geolevel2)

    # Manual Fix GEOLEVEL1 for some countries
    # Thailand, Colombia, Ecuador, Lesoto and Mauritius
    # gdf.loc[gdf.CNTRY_NAME == "Thailand", "GEOLEVEL1"] = gdf.loc[gdf.CNTRY_NAME == "Thailand", "GEOLEVEL2"].str[:6]
    
    id_cols = ["CNTRY_CODE", "GEOLEVEL1", "GEOLEVEL2"] 
    gdf = gdf[["geometry"] + id_cols]
    # AUTOMATED FIXES    
    for col in id_cols:
        gdf[col] = pd.to_numeric(gdf[col]).fillna(0)
    countries_with_no_dissagregation = [246, 348, 528]
    gdf = gdf[~gdf.GEOLEVEL2.isin(countries_with_no_dissagregation)] # Drop countries with no disagregation
    gdf = fix_IPUMS_missing_geolevels(gdf)
    gdf = fix_IPUMS_invalid_geolevel1(gdf)
    gdf = fix_IPUMS_conflicting_international_boundaries(wb_map, gdf, export_maps=True)

    # Drop unavailable data
    gdf = gdf[gdf.GEOLEVEL1 != 0] 
    gdf = gdf[gdf.GEOLEVEL2 != 888888888]
    gdf = gdf.dropna(subset=id_cols)

    assert gdf.duplicated(subset=id_cols).sum() == 0, "There are duplicated rows in the data!"
    
    # Create ID
    gdf["ID"] = gdf.groupby(id_cols).ngroup()
    assert gdf.ID.nunique() == gdf.shape[0], "ID is not unique!, there's some bug in the code..."
    print("Data loaded!")
    return gdf

def open_IPUMS(force_geolev1):
    path = rf"{DATA_RAW}\IPUMS Fixed"
    files = os.listdir(path)
    shpfiles = [f for f in files if f.endswith(".shp")]
    geo2_files = [f for f in shpfiles if "geo2_" in f]
    geo2_files = [f for f in geo2_files if f.split("_")[1][:2] not in force_geolev1] # Remove countries that have to be aggregated. See issue #35
    geo2_countries = [name.split("_")[1][:2] for name in geo2_files]
    geo1_files = [f for f in shpfiles if "geo1_" in f]
    geo1_files = [f for f in geo1_files if f.split("_")[1][:2] not in geo2_countries]
    # return geo1_files, geo2_files
    files = geo2_files + geo1_files
    countries = [name.split("_")[1][:2] for name in files]
    countries = pd.Series(countries)
    assert countries.duplicated().sum() == 0, f"There are duplicated countries in the data: {countries[countries.duplicated()]}!"
    
    gdfs = []
    file_created = False
    schema = None
    pqwriter = None
    for f in tqdm(files):
        gdf = gpd.read_file(os.path.join(path, f))
        
        # Store in parquet
        pqwriter, file_created, schema = store_in_parquet_file(gdf, pqwriter, file_created, schema)
        # Guardo el chunk en un archivo parquet

        ctry_name  = gdf.CNTRY_NAME[0]
        
        # MANUAL FIXES
        if ctry_name == "United States":
            print("Removing Puerto Rico from US...")
            gdf = gdf[~gdf.GEOLEVEL1.isin(["840721"])]
            assert gdf.shape[0] > 0
        if ctry_name == "Nigeria":
            print("Removing glitch from Nigeria...")
            gdf = gdf[gdf.GEOLEVEL2 != "566024008"]
            assert gdf.shape[0] > 0

        gdfs += [gdf]
        
    pqwriter.close()
    gdf = gpd.GeoDataFrame(pd.concat(gdfs))
    gdf = gdf.set_crs("EPSG:4326")
    
    return gdf

def load_WB_country_data():
    print("Loading World Bank country data...")
    WB_country = gpd.read_file(rf"{DATA_RAW}\world_bank_adm2\world_bank_adm2.shp")
    
    # Assign nan when ADM2 is not available 
    WB_country.loc[WB_country.ADM2_NAME == "Administrative unit not available", "ADM2_CODE"] = (
        np.nan
    )
    
    # Create ADM_LAST variable: ADM2_NAME if available, else ADM1_NAME
    for col in ["ADM0_CODE", "ADM1_CODE", "ADM2_CODE"]:
        WB_country[col] = WB_country[col].fillna(0)

    # Dissolve by ADM_LAST and country code    
    WB_country = WB_country.dissolve(by=["ADM2_CODE","ADM1_CODE", "ADM0_CODE"]).reset_index()
    
    # # Create ID
    WB_country["ID"] = WB_country.groupby(["ADM2_CODE", "ADM1_CODE", "ADM0_CODE"]).ngroup()
    assert WB_country.ID.nunique() == WB_country.shape[0], "ID is not unique!, there's some bug in the code..."
    print("Data loaded!")
    return WB_country

def rasterize_shape_like_dataset(shape, dataset):
    print("Rasterizing shape...")
    raster = make_geocube(
        vector_data=shape,
        like=dataset,
    )
    # For some reason, like option is not working, so I have to manually add x and y
    assert (raster["x"].shape == dataset["x"].shape)
    assert (raster["y"].shape == dataset["y"].shape)
    raster["x"] = dataset["x"]
    raster["y"] = dataset["y"]
    raster = raster.drop_vars(["spatial_ref"])
    print("Done!")
    return raster

def store_in_parquet_file(gdf, pqwriter, file_created, schema):
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet_file_path = fr"{DATA_PROC}\IPUMS_full.parquet"
    df = pd.DataFrame(gdf.copy())
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt if x is not None else None)
    
    if "GEOLEVEL1" not in df.columns:
        df["GEOLEVEL1"] = "None"
    elif "GEOLEVEL2" not in df.columns:
        df["GEOLEVEL2"] = "None"
        
    table = pa.Table.from_pandas(df)
    if file_created is False:
        # create a parquet write object giving it an output file
        schema = table.schema
        pqwriter = pq.ParquetWriter(parquet_file_path, schema)
        file_created = True
    else:
        # Ensure the columns are in the same order as the initial schema
        df = df[[field.name for field in schema]]
        table = pa.Table.from_pandas(df)

    pqwriter.write_table(table)
    
    return pqwriter, file_created, schema


def fix_IPUMS_conflicting_international_boundaries(wb, ipums, export_maps=False):
    ''' Fix IPUMS conflicting international boundaries by clipping the data to the World Bank boundaries.
    
    Parameters:
    wb (GeoDataFrame): World Bank boundaries
    ipums (GeoDataFrame): IPUMS boundaries
    
    Returns:
    GeoDataFrame: IPUMS boundaries with the conflicting boundaries clipped to the World Bank boundaries
    '''
    print("Normalizando límites internacionales...")
    countries_to_clip = {
        # Countries with conflicting boundaries
        "Marruecos": {"WB": 169, "IPUMS": 504},
        "South Sudan": {"WB": 74, "IPUMS": 728},
        "Sudan": {"WB": 6, "IPUMS": 729},
        "Egypt": {"WB": 40765, "IPUMS": 818},
        "Kenya": {"WB": 133, "IPUMS": 404},
        "Russia": {"WB": 204, "IPUMS": 643},
        "India": {"WB": 115, "IPUMS": 356},
        "China": {"WB": 147295, "IPUMS": 156},
        "Kyrghyzstan": {"WB": 138, "IPUMS": 417},
    }                        

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for country, codes in countries_to_clip.items():
            wbcode = codes["WB"]
            ipumscode = codes["IPUMS"]
            
            # Clip IPUMS using WB
            clipped = (
                ipums[ipums.CNTRY_CODE == ipumscode]
                .clip(wb[wb.ADM0_CODE == wbcode])
            )
            
            if export_maps:
                # Plot the clipped data
                fig, ax = plt.subplots(figsize=(10, 10))
                ipums[ipums.CNTRY_CODE == ipumscode].plot(ax=ax)
                clipped.plot(ax=ax, facecolor="none", edgecolor="red")
                plt.savefig(f"{DATA_PROC}/fixes/{country}.png")
                plt.close()

            # remove areas with residual geometry
            clipped = clipped[clipped.geometry.area > .0001]
            clipped = clipped[clipped.geometry.is_valid]
            print(f"  -  {country}: Se eliminaron ", len(ipums[ipums.CNTRY_CODE == ipumscode]) - len(clipped), " registros")
            
            # Update the original dataframe
            ipums.loc[ipums.CNTRY_CODE == ipumscode] = clipped
            
        # Remove unwanted shapes from Israel & Palestine
        ipums = ipums[~(ipums["GEOLEVEL2"].astype(str).str[-2:].isin(["97", "98", "99"]) & ipums["CNTRY_CODE"].isin([376, 275]))]
        # Clean empty registries from the dataframe
        ipums = ipums[ipums.geometry.is_valid]

    return ipums

def fix_IPUMS_missing_geolevels(gdf):
    ''' Fix missing geolevels in IPUMS data by filling the missing geolevels with the other geolevel.  '''
    # Fix GEOLEVEL1 and GEOLEVEL2 (fill geo2 with geo1 if missing and viceversa)
    gdf.loc[gdf["GEOLEVEL1"]==0, "GEOLEVEL1"] = gdf.loc[gdf["GEOLEVEL1"]==0, "GEOLEVEL2"].astype(str).str[:6].astype(int) 
    gdf.loc[gdf["GEOLEVEL2"]==0, "GEOLEVEL2"] = gdf.loc[gdf["GEOLEVEL2"]==0, "GEOLEVEL1"] 

    return gdf
    
def fix_IPUMS_invalid_geolevel1(gdf):
    ''' For some countries, some geolevel2 geometries are assigned to different geolevel1. 
        This function fixes the geolevel1 for those countries. See issue #35 for more details.
    '''
    countries_to_fix_geolevel1 = [
        764, # Tailandia
        170, # Colombia
        218, # Ecuador
        480, # Mauricio
    ]
    for country in countries_to_fix_geolevel1:
        gdf.loc[gdf.CNTRY_CODE == country, "GEOLEVEL1"] = gdf.loc[gdf.CNTRY_CODE == country, "GEOLEVEL2"].astype(str).str[:6].astype(int)
    
    return gdf    

# def fix_IPUMS_invalid_geolevel2(gdf, countries):
#     ''' For some countries, the geolevel2 is not consistent across censuses. This function fixes the geolevel2 for those countries,
#         by reducing the aggregation level to geolevel1. See issue #35 for more details.
    
#     '''
#     for country in countries:
#         gdf.loc[gdf.CNTRY_CODE == country, "GEOLEVEL2"] = gdf.loc[gdf.CNTRY_CODE == country, "GEOLEVEL2"].astype(str).str[:6].astype(int)

#     return gdf