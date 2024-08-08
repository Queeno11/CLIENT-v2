import re
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import psutil

try:
    import cupy as cp
    from cupyx.scipy.interpolate import RegularGridInterpolator
except:
    pass
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
from tqdm import tqdm
from rasterio.windows import Window
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


def compute_zonal_statistics(datavar, adm_id, year_gpw):
    """Compute zonal statistics using CuPy.

    Parameters:
    -----------
    datavar: xarray.DataArray
        Data to be aggregated.
    adm_id: xarray.DataArray
        Administrative data to group the data.
    year_gpw: xarray.DataArray
        Population data to weight the aggregation.

    Returns:
    --------
    pandas.DataFrame: Zonal statistics aggregated by administrative unit.
    """
    groups = cp.asarray(adm_id.values.flatten())
    population_values = cp.asarray(year_gpw.flatten(), dtype=np.float32)
    area_values = cp.asarray(datavar.flatten())

    assert (
        groups.shape == area_values.shape == population_values.shape
    ), f"There's something wrong with the shapes of the data, they all must match: {groups.shape}, {area_values.shape}, {population_values.shape}"

    # Filter out nan values
    mask = (groups != 99999) & ~cp.isnan(area_values)
    groups = groups[mask]
    area_values = area_values[mask]
    population_values = population_values[mask]

    # If population is missing, set it to 0
    population_values = cp.nan_to_num(population_values)

    # Calculate the weighted sum of affected area and population
    n_total_area = groupby_sum_cupy(groups, values=None)
    n_affected_area = groupby_sum_cupy(groups, values=area_values)
    n_total_pop = groupby_sum_cupy(groups, values=population_values)
    n_affected_pop = groupby_sum_cupy(groups, values=population_values * area_values)

    cols = {
        "cells_affected": n_affected_area,
        "total_cells": n_total_area,
        "population_affected_n": n_affected_pop,
        "total_population": n_total_pop,
    }
    df = pd.DataFrame()
    for colname, coldata in cols.items():
        df = df.join(
            pd.DataFrame.from_dict(coldata, orient="index", columns=[colname]),
            how="outer",
        )

    return df


def groupby_sum_cupy(groups, values=None):
    """Calculate the weighted sum of values based on the groups.

    Parameters:
    -----------
    groups: cp.array of shape (n, 1)
        List of group indices.
    values: cp.array of shape (n, 1) (optional)
        List of values to sum. If None, the count for each group is calculated.

    Returns:
    --------
    dict: Dictionary with the group indices as keys and the sum as values.
    """
    n_total_area = cp.bincount(groups, weights=values)
    n_total_area = cp.nan_to_num(n_total_area)
    unique_values = cp.arange(len(n_total_area))[n_total_area > 0]
    result = dict(
        zip(cp.asnumpy(unique_values), cp.asnumpy(n_total_area[unique_values]))
    )

    return result


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
    lat = cp.asarray(era5_da.y.values, dtype="float32")
    lon = cp.asarray(era5_da.x.values, dtype="float32")
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

    # If the data is not in the correct shape, crop or interpolate
    has_cropped_shape = all(x == y for x, y in zip(shock_dims, cropped_shape))
    has_bigger_shape = any(x > y for x, y in zip(shock_dims, full_30arc_sec_shape))
    has_smaller_shape = any(x < y for x, y in zip(shock_dims, full_30arc_sec_shape))

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
    f = f.replace("WB_", "").replace("IPUMS_", "")
    f = f.split("_")
    assert shockname in [
        "drought",
        "floods",
        "hurricanes",
        "heatwaves",
        "coldwaves",
        "intenserain",
    ], "Invalid shockname. If you are adding a new shockname, please add it to the list in utils.parse_filename()."
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
            "threshold": "",
            "year": f[2],
            "chunk": f[3],
        }
    elif shockname == "hurricanes":
        return {
            "variable": f"{f[1]} {f[2]}",  # ej. category 5
            "threshold": f[3],  # buffer size, ej. b50
            "year": f[4],
            "chunk": f[5],
        }
    elif shockname == "heatwaves":
        return {
            "variable": f"{f[1][:2]}",  # ej. category 5
            "threshold": f[1][2:],  # split 4025 in 40 and 25
            "year": f[2],
            "chunk": f[3],
        }
    elif shockname == "coldwaves":
        return {
            "variable": f"{f[1][:2]}",  # ej. category 5
            "threshold": f[1][2:],  # split 4025 in 40 and 25
            "year": f[2],
            "chunk": f[3],
        }
    elif shockname == "intenserain":
        return {
            "variable": f[1],  # rx1day
            "threshold": f[2],  # 200
            "year": f[3],
            "chunk": f[4],
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
    df = df.reset_index().rename(columns={"index": "ID"})
    df["threshold"] = df["threshold"]
    df["variable"] = df["variable"]
    return df


def parse_columns(names: tuple):
    agg = names[0]
    string = names[1]
    letter = agg[0]

    return f"{string}_{letter}"


def compress_dataframe(df):
    "" "Compress the dataframe to save memory" ""

    df["cells_affected"] = df["cells_affected"].fillna(0).astype(np.uint32)
    df["total_cells"] = df["total_cells"].fillna(0).astype(np.uint32)
    df["population_affected_n"] = (
        df["population_affected_n"].fillna(0).astype(np.uint64)
    )
    df["total_population"] = df["total_population"].fillna(0).astype(np.uint64)
    return df


def process_all_dataframes(gdf, parquet_paths, shockname):
    import gc

    pd.set_option("future.no_silent_downcasting", True)

    gdf.columns = [col.lower() if col != "ID" else col for col in gdf.columns]

    files = os.listdir(parquet_paths)
    files = [f for f in files if f.endswith(".parquet") and shockname in f]
    files = [f for f in files if "out_" not in f]
    outpath = os.path.join(parquet_paths, f"out_{shockname}_ungrouped.parquet")

    print("Reading and concatenating dataframes...")
    # Memory efficient way to concatenate dataframes:
    #   https://www.confessionsofadataguy.com/solving-the-memory-hungry-pandas-concat-problem/
    pqwriter = None
    create_file = True
    for f in tqdm(files):
        try:
            df = pd.read_parquet(os.path.join(parquet_paths, f))
            df = compress_dataframe(df)

            # Agrego como cols la variable, threshold, year y chunk
            names = parse_filename(f, shockname)
            for col, value in names.items():
                df[col] = value

            # Proceso el chunk
            df = process_chunk(df)

            # Guardo el chunk en un archivo parquet
            table = pa.Table.from_pandas(df)
            if create_file:
                # create a parquet write object giving it an output file
                pqwriter = pq.ParquetWriter(outpath, table.schema)
                create_file = False
            pqwriter.write_table(table)

        except Exception as e:
            print(f"Error with file {f}: {e}")

    if pqwriter:
        pqwriter.close()

    gc.collect()
    print(f"Se creó {outpath}")
    df = pd.read_parquet(outpath)
    print("Grouping data...")
    df = df.groupby(["ID", "year", "variable", "threshold"]).sum()
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
    return df


def process_to_stata(df, gdf, parquet_paths, shockname):
    df["name"] = df["variable"].str.lower() + "_" + df["threshold"].astype(str)
    df = df.drop(
        columns=[
            "variable",
            "threshold",
            "chunk",
        ]
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

    path = os.path.join(parquet_paths, f"{shockname}_wide.csv")
    pivot.to_csv(path)
    print(f"Se creó {path}")

    pivot = pd.read_csv(path)

    # Add the data to the gdf
    out_df = gdf.merge(pivot, on="ID", validate="1:m", how="outer")
    out_df.columns = out_df.columns.str.lower()

    return out_df


def coordinates_from_0_360_to_180_180(ds):
    ds["x"] = ds.x.where(ds.x < 180, ds.x - 360)
    ds = ds.sortby("x")
    return ds


def try_loading_ds(ds):
    memory_required = ds.nbytes
    available_memory = psutil.virtual_memory().available
    if memory_required < available_memory:
        # print("Loading shock in memory...")
        ds = ds.load()
        is_loaded = True
    else:
        is_loaded = False
    return ds, is_loaded
