"""
This code creates functions to compute the Standardized Precipitation Index (SPI), Standardized
 Precipitation Evapotranspiration Index (SPEI) and Potential Evapotranspiration (PET). It checks if the
 processed data file exists and, if not, it compiles yearly data files into a single dataset, processes the
 data calculating PET, SPI and SPEI for various timescales and saves the results to a NetCDF file.

Input: ERA5_monthly data

Output:
- ERA5_monthly_1970-2021_SPI-SPEI.nc
- ERA5_monthly_1970-2021_withPET.nc
"""

"""
Packages used

- numpy: package for scientific computing in Python. It supports large,
multi-dimensional arrays and matrices, along with a collection of mathematical functions
to operate on these arrays.
- xarray: package for working with labeled multi-dimensional arrays and datasets
- pandas: package for data manipulation and analysis
- geopandas: extension of pandas that makes working with geospatial data
- matplotlib: A 2D plotting library in Python. It is used to create static, animated, and interactive visualizations.
- seaborn: A data visualization library based on matplotlib
- dask.diagnostics: tools to diagnose and visualize the progress of Dask tasks
- climate_indices: A package for calculating climate indices such as the Standardized Precipitation Index (SPI)
and the Standardized Precipitation Evapotranspiration Index (SPEI).
- warnings: module for controlling Python's warning messages
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from climate_indices import indices, compute
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Path Definition
ERA5_DATA = rf"Z:\WB Data\ERA5 Reanalysis\monthly-land"
PATH = "Z:\Laboral\World Bank\CLIENT v2"
DATA_RAW = rf"{PATH}\Data\Data_raw"
DATA_PROC = rf"{PATH}\Data\Data_proc"
DATA_OUT = rf"{PATH}\Data\Data_out"

# Parameters
distribution = indices.Distribution.gamma
data_start_year = 1970
calibration_year_initial = 1970
calibration_year_final = 2020
periodicity = compute.Periodicity.monthly

#################################
####        Functions       #####
#################################


def spi_wrapper(
    t2m,
    scale,
    distribution,
    data_start_year,
    calibration_year_initial,
    calibration_year_final,
    periodicity,
):
    t2m_copy = np.copy(t2m)  # Make a writable copy of t2m
    return indices.spi(
        t2m_copy,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
    )


# Define the PET function to be vectorized
def pet_wrapper(t2m, lat, year):
    t2m_copy = np.copy(t2m)  # Make a writable copy of t2m
    return indices.pet(t2m_copy, lat, year)


def spei_wrapper(
    t2m,
    pet,
    scale,
    distribution,
    data_start_year,
    calibration_year_initial,
    calibration_year_final,
    periodicity,
):
    t2m_copy = np.copy(t2m)  # Make a writable copy of t2m
    pet_copy = np.copy(pet)  # Make a writable copy of pet
    return indices.spei(
        t2m_copy,
        pet_copy,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
    )


#################################
#### Load ERA5 monthly data #####
#################################

if not os.path.exists(os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_withPET.nc")):

    ## Compile data by year
    print("Creating ERA5 monthly time series...")
    files = os.listdir(ERA5_DATA)
    datasets = []
    for file in files:
        ds = xr.open_dataset(
            os.path.join(ERA5_DATA, file), chunks={"latitude": 500, "longitude": 500}
        )
        datasets += [ds]
    ds = xr.concat(datasets, dim="time")
    print("Saving compiled data...")
    with ProgressBar():
        ds.to_netcdf(os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_noPET.nc"))

    ## Process compiled data
    ds = xr.open_dataset(
        os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_noPET.nc"),
        chunks={"latitude": 500, "longitude": 500},
    )
    # Remove -90 and 90 bounds, it produces errors in the PET calculation
    ds = ds.sel(latitude=slice(89.9, -89.9))
    # ds = ds.where(mask)
    # ds = ds.sel(
    #     latitude=slice(-10.660608, -47.872144), longitude=slice(111.621094, 181.582031)
    # )

    # Change from Kelvin to Celsius
    ds["t2m"] = ds["t2m"] - 273.15

    ds["PET"] = xr.apply_ufunc(
        pet_wrapper,
        ds.t2m,
        ds.latitude,
        1970,
        vectorize=True,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",  # Enable Dask for parallel execution, if using Dask arrays
        output_dtypes=[np.float64],
    )

    print("Saving processed data...")

    with ProgressBar():
        ds.to_netcdf(
            os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_withPET.nc"),
        )
    try:
        os.remove(os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_noPET.nc"))
    except:
        pass

#################################
####       Compute SPI      #####
#################################

### Running this takes... A lot. Aprox. 90m for each SPI, so ~7.5h for all SPIs.

## Script based on: https://github.com/monocongo/climate_indices/issues/326
## Original paper: https://www.droughtmanagement.info/literature/AMS_Relationship_Drought_Frequency_Duration_Time_Scales_1993.pdf
## User guide to SPI: https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1208&context=droughtfacpub
#   It is recommended to use SPI-9 or SPI-12 to compute droughts.
#   "SPI values below -1.5 for these timescales (SPI-9) are usually a good indication that dryness is having a significant impact on
#    agriculture and may be affecting other sectors as well."
## More here: https://www.researchgate.net/profile/Sorin-Cheval/publication/264467702_Spatiotemporal_variability_of_the_meteorological_drought_in_Romania_using_the_Standardized_Precipitation_Index_SPI/links/5842d18a08ae2d21756372f8/Spatiotemporal-variability-of-the-meteorological-drought-in-Romania-using-the-Standardized-Precipitation-Index-SPI.pdf
## Ignore negative values, they are normal: https://confluence.ecmwf.int/display/UDOC/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ


print(f"Computing SPI and SPEI")
ds = xr.open_dataset(
    os.path.join(DATA_PROC, "ERA5_monthly_1970-2021_withPET.nc"),
    chunks={"latitude": 500, "longitude": 500},
)

out_path = os.path.join(DATA_OUT, f"ERA5_monthly_1970-2021_SPI-SPEI.nc")

for i in [1, 3, 6, 12, 24]:

    ds[f"SPI-{i}"] = xr.apply_ufunc(
        spi_wrapper,
        ds.tp,
        i,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        vectorize=True,
        input_core_dims=[["time"], [], [], [], [], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",  # Enable Dask for parallel execution, if using Dask arrays
        output_dtypes=[np.float64],
    )

    ds[f"SPEI-{i}"] = xr.apply_ufunc(
        spei_wrapper,
        ds.tp,
        ds.PET,
        i,
        distribution,
        periodicity,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        vectorize=True,
        input_core_dims=[["time"], ["time"], [], [], [], [], [], []],
        output_core_dims=[["time"]],
        dask="parallelized",  # Enable Dask for parallel execution, if using Dask arrays
        output_dtypes=[np.float64],
    )

with ProgressBar():
    ds.to_netcdf(out_path)
