import time
import cupy as cp
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window
from cupyx.scipy.interpolate import RegularGridInterpolator
from decorator import decorator
from line_profiler import LineProfiler


@decorator
def profile_each_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        profiled_func(*args, **kwargs)
    finally:
        profiler.print_stats()


# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import seaborn as sns


def compute_zonal_statistics(groups, area_values, population_values, sorted=False):
    """Groupby mean function using cupy. Data doesn't have to be sorted.

    Parameters:
    -----------
    groups: cp.array of shape (n, 1)
        Data to be grouped by the first column.
    values: cp.array of shape (n, 1)
        Data to be averaged according to groups.

    """
    # Remove rows with NaNs
    mask = ~cp.isnan(groups) & ~cp.isnan(area_values)
    groups = groups[mask]
    area_values = area_values[mask]
    population_values = population_values[mask]
    del mask

    # Sort based on the groups if not already sorted
    if not sorted:
        sort_indices = cp.argsort(groups)
        groups = groups[sort_indices]
        area_values = area_values[sort_indices]
        population_values = population_values[sort_indices]
        del sort_indices

    _id, _pos, g_count = cp.unique(groups, return_index=True, return_counts=True)
    del groups

    g_area_sum = cp.add.reduceat(area_values, _pos)
    g_pop_sum = cp.add.reduceat(population_values, _pos)
    g_affected_pop_sum = cp.add.reduceat(population_values * area_values, _pos)
    del area_values

    g_area_mean = g_area_sum / g_count
    g_pop_mean = g_affected_pop_sum / g_pop_sum
    aggs = dict(
        zip(
            cp.asnumpy(_id),
            zip(
                cp.asnumpy(g_area_mean),
                cp.asnumpy(g_area_sum),
                cp.asnumpy(g_count),
                cp.asnumpy(g_pop_mean),
                cp.asnumpy(g_affected_pop_sum),
                cp.asnumpy(g_pop_sum),
            ),
        )
    )
    # print(g_area_mean)
    # print(g_pop_mean)
    # print(aggs)
    return aggs


# @profile_each_line
def get_interpolate_xy(era5_da, adm_id_da):

    # Define the grid
    lat = cp.asarray(era5_da.y.values)
    lon = cp.asarray(era5_da.x.values)
    era5_original_grid = (lat, lon)

    # New grid based on adm_id_da
    new_lat = cp.asarray(adm_id_da.y)
    new_lon = cp.asarray(adm_id_da.x)
    era5_new_grid = np.meshgrid(new_lat, new_lon, indexing="ij")

    return era5_original_grid, era5_new_grid


def intepolate_era5_data(era5_da, adm_id_da, verbose=False):
    """Interpolate ERA5 data to the adm_id_da grid using cupy and cupyx

    Parameters:
    -----------
    era5_da: xarray.DataArray
        ERA5 data to be interpolated.
    adm_id_da: xarray.DataArray
        DataArray with the grid to interpolate the ERA5 data.
    verbose: bool
        If True, print the time it takes to interpolate the data.

    Returns:
    --------
    era5_interp: cupy.array
        Interpolated ERA5 data to the adm_id_da grid.
    """
    if verbose:
        print("Interpolating...")
        t0 = time.time()

    ## Interpolate droughts like adm_id_da
    # Define the grid
    lat = cp.asarray(era5_da.y.values)
    lon = cp.asarray(era5_da.x.values)
    spi = cp.asarray(era5_da.values, dtype="bool")
    interpolate = RegularGridInterpolator(
        (lat, lon), spi, method="nearest", bounds_error=False
    )

    # Interpolate
    new_lat = cp.asarray(adm_id_da.y)
    new_lon = cp.asarray(adm_id_da.x)
    new_lat, new_lon = np.meshgrid(new_lat, new_lon, indexing="ij")
    era5_interp = interpolate((new_lat, new_lon))
    print(era5_interp.shape)
    if verbose:
        t1 = time.time()
        print(f"Time Interp (cupyx.scipy): {t1 - t0}")

    return era5_interp.astype("bool")
