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

    if verbose:
        t1 = time.time()
        print(f"Time Interp (cupyx.scipy): {t1 - t0}")

    return era5_interp.astype("bool")
<<<<<<< Updated upstream
=======


def get_bounds_from_chunk_number(chunk_number, total_chunks=8, canvas=None):
    """Get the bounding box coordinates for a given chunk number.

    Data is divided into total_chunks chunks, each covering an 1/total_chunks of the globe.

    Parameters:
    -----------
    chunk_number: int
        Chunk number < total_chunks.
    total_chunks: int
        Total number of chunks to divide the globe into.
    canvas: tuple (optional)
        Bounding box coordinates (left, bottom, right, top) to divide the globe into chunks.

    Returns:
    --------
    tuple: Bounding box coordinates (left, bottom, right, top).
    """

    if chunk_number > total_chunks - 1:
        raise ValueError("Chunk number must be less than total_chunks.")

    # Define the bounding box coordinates for each chunk
    if canvas is None:
        x_min, y_min, x_max, y_max = -180, -90, 180, 90
    else:
        x_min, y_min, x_max, y_max = canvas
    # Calculate the bounding box coordinates for the given chunk number
    side_chunks = np.sqrt(total_chunks)
    if not side_chunks.is_integer():
        raise ValueError("Total chunks must be a square number.")
    side_chunks = int(side_chunks)

    chunk_position = np.unravel_index(chunk_number, (side_chunks, side_chunks))

    x_step = (x_max - x_min) / side_chunks
    y_step = (y_max - y_min) / side_chunks

    left = x_min + chunk_position[0] * x_step
    right = left + x_step
    bottom = y_min + chunk_position[1] * y_step
    top = bottom + y_step

    return (left, bottom, right, top)


def get_filter_from_chunk_number(chunk_number, total_chunks=8, canvas=None):
    """Get the filter for a given chunk number.

    Data is divided into total_chunks chunks, each covering an 1/total_chunks of the globe.

    Example:
    filter = get_filter_from_chunk_number(1, total_chunks=8)
    filtered_ds = ds.sel(filter)

    Parameters:
    -----------
    chunk_number: int
        Chunk number < total_chunks.
    total_chunks: int
        Total number of chunks to divide the globe into.

    Returns:
    --------
    filter: dictionary to filter the xarray.Dataset.
    chunk_bounds: tuple of the bounding box coordinates (left, bottom, right, top).

    """

    chunk_bounds = get_bounds_from_chunk_number(
        chunk_number, total_chunks=total_chunks, canvas=canvas
    )
    filter = dict(
        x=slice(chunk_bounds[0], chunk_bounds[2]),
        y=slice(chunk_bounds[3], chunk_bounds[1]),
    )
    return filter, chunk_bounds


def find_gpw_closes_year(year):
    """Find the closest year in the GPW dataset to the given year.

    GPW data comes from 2000, 2005, 2010, 2015 and 2020.

    Parameters:
    -----------
    year: int
        Year to find the closest GPW data.

    Returns:
    --------
    int: Closest year in the GPW dataset.
    """

    years = [2000, 2005, 2010, 2015, 2020]
    return min(years, key=lambda x: abs(x - year))


def load_gpw_data(year, bounds=None):
    """Load the GPW data for given year as a CuPy array.

    If bounds are provided, the data is clipped to the bounding box.

    Parameters:
    -----------
    year: int
        Year of the GPW data to load.
    bounds: tuple
        Bounding box coordinates (left, bottom, right, top) to load the data.
    """
    if year not in [2000, 2005, 2010, 2015, 2020]:
        raise ValueError("Year must be one of 2000, 2005, 2010, 2015 or 2020.")

    with rasterio.open(
        rf"/mnt/d/datasets/Gridded Population of the World/gpw_v4_population_count_rev11_{year}_30_sec.tif",
    ) as src:
        window = None
        if bounds is not None:
            # Read the data into a NumPy array
            window = rasterio.windows.from_bounds(
                left=bounds[0],
                bottom=bounds[1],
                right=bounds[2],
                top=bounds[3],
                transform=src.transform,
            )
        data = src.read(1, window=window)  # Assuming single band, read the first band
        # Verify we've loaded data
        assert data is not None, "No data loaded from the GPW raster."

        # Convert the NumPy array to a CuPy array
        gpw = cp.asarray(data)

    return gpw


def identify_needed_transformations(shock, adm_data):
    """Identify if the data needs to be cropped or interpolated to match
    administrative/population data.

    If the data has a lower resolution than the administrative data, it
    will be interpolated to match the amd shape. If the data the same resolution
    but the bounds cover the entire globe, it will be cropped to match the adm shape.

    Parameters:

    shock: xarray.DataArray
        Data to be transformed.
    adm_data: xarray.DataArray
        Administrative data to match the shape of the shock data.

    Returns:
    --------
    xarray.DataArray: Transformed shock data.
    bool: True if the data needs to be interpolated.
    bool: True if the data needs to be coarsened.
    """

    # Crop data to x-y limits
    xmin, ymin, xmax, ymax = (
        adm_data.x.min(),
        adm_data.y.min(),
        adm_data.x.max(),
        adm_data.y.max(),
    )
    shock = shock.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    # Query the dimensions of the data
    shock_dims = tuple(shock.sizes[d] for d in ["x", "y"])
    full_30arc_sec_shape = (43200, 21600)
    cropped_shape = tuple(adm_data.sizes[d] for d in ["x", "y"])

    print(shock_dims, full_30arc_sec_shape, cropped_shape)
    # If the data is not in the correct shape, crop or interpolate
    has_cropped_shape = all(x == y for x, y in zip(shock_dims, cropped_shape))
    has_bigger_shape = any(x > y for x, y in zip(shock_dims, full_30arc_sec_shape))
    has_smaller_shape = any(x < y for x, y in zip(shock_dims, full_30arc_sec_shape))
    print(has_bigger_shape, has_cropped_shape, has_smaller_shape)
    needs_crop = has_bigger_shape
    needs_interp = (
        (not has_cropped_shape) and (not has_bigger_shape) and has_smaller_shape
    )
    need_coarsen = (
        (not has_cropped_shape) and (not has_bigger_shape) and (not has_smaller_shape)
    )

    if need_coarsen:
        raise NotImplementedError(
            "Coarsening is not implemented yet. Reduce the x, y dimensions of the data to match the Gridded Population of the World 30arc-sec grid before running this script."
        )
    if needs_crop:
        print("Data will be cropped to match the administrative/GPW data...")
        shock = shock.sel(
            x=slice(adm_data.x.min(), adm_data.x.max()),
            y=slice(adm_data.y.max(), adm_data.y.min()),
        )
    if needs_interp:
        print("Data will be interpolated to match the administrative/GPW data...")

    return shock, needs_interp, need_coarsen


def parse_filename(f, shockname):
    """Parse the filename to extract the variable, threshold, year and chunk

    Parameters:
    ----------
    f : str
        Filename to parse

    Returns:
    -------
    dict
        Dictionary with the following keys:
            - variable: str
            - threshold: str
            - year: str
    """

    f = f.split("_")
    if shockname == "drought":
        return {
            "variable": f[1],
            "threshold": f"{f[2]}",
            "year": f[3],
            "chunk": f[4],
        }
    elif shockname == "floods":
        return {
            "variable": f[1],
            "year": f[2],
            "chunk": f[3],
            "threshold": "",
        }


def process_chunk(df):
    """Process the dataframe generated from zonal_statistics

    Parameters:
    ----------
    df : pd.DataFrame
        Dataframe to process

    Returns:
    -------
    df : pd.DataFrame
        Processed dataframe
    """
    df = df.reset_index(names=["ID"])
    df["threshold"] = df["threshold"]
    df["variable"] = df["variable"]
    df["name"] = df["variable"].str.lower() + df["threshold"].astype(str)
    df = df.drop(
        columns=[
            "variable",
            "threshold",
            "chunk",
        ]
    )
    return df


def parse_columns(names: tuple):
    agg = names[0]
    string = names[1]
    letter = agg[0]

    return f"{string}_{letter}"


def process_all_dataframes(gdf, parquet_paths, shockname):
    import dask.dataframe as dd
    
    gdf.columns = [col.lower() for col in gdf.columns]

    files = os.listdir(parquet_paths)
    files = [f for f in files if f.endswith(".parquet") and shockname in f]

    dfs = []
    for f in tqdm(files):
        df = dd.read_parquet(os.path.join(parquet_paths, f))
        # Agrego como cols la variable, threshold, year y chunk
        names = parse_filename(f, shockname)
        df[list(names.keys())] = list(names.values())
        # Proceso el chunk
        dfs += [process_chunk(df)]

    # Concatenate all the dataframes and create the shock variables
    df = pd.concat(dfs)

    df = df.groupby(["ID", "name", "year"]).sum()
    df["area_affected"] = df["cells_affected"] / df["total_cells"]
    df["population_affected"] = df["population_affected_n"] / df["total_population"]
    df = (
        df.drop(
            columns=[
                "cells_affected",
                "total_cells",
                "population_affected_n",
                "total_population",
            ]
        )
        .reset_index()
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )

    # Pivot data: every shock has to be a column
    pivot = df.pivot(
        index=["ID", "year"],
        columns="name",
        values=["population_affected", "area_affected"],
    )

    # Reindex the two-level columns pivot returns
    newcols = []
    for cols in pivot.columns:
        newcols += [parse_columns(cols)]
    pivot.columns = newcols
    pivot = pivot.reset_index()

    # Add the data to the gdf
    out_df = gdf.merge(pivot, left_on="id", right_on="ID", validate="1:m", how="outer")
    out_df.rename(
        columns={
            "adm0_code": "adm0",
            "admlast_code": "adm_lst",
        },
        inplace=True,
    )

    return out_df, newcols


def coordinates_from_0_360_to_180_180(ds):
    ds["x"] = ds.x.where(ds.x < 180, ds.x - 360)
    ds = ds.sortby("x")
    return ds
>>>>>>> Stashed changes
