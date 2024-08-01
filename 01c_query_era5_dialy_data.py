"""
The code automates the download of ERA5 reanalysis data for each year from 1970 to 2020.
It checks if the data file for each year already exists in the specified folder, and if not,
it uses the Copernicus Climate Data Store (CDS) API to download the data.

Input: none

Output: data_{year}.netcdf.zip from 1970 to 2020
"""

"""
Packages used

- os: provides a way of using operating system-dependent functionality, such as reading or writing to the file system.
- cdsapi: part of the Climate Data Store API (CDS API). It is used to access climate change.
- tqdm: displaying progress bars in Python
"""
import os
import cdsapi
from tqdm import tqdm

folder = "/mnt/d/Datasets/ERA5 Reanalysis/dialy-single-levels"
downloaded = os.listdir(folder)
downloaded = [int(x.split("_")[1].split(".")[0]) for x in downloaded]
last_year = max(downloaded) if downloaded else (2020 - 50)

c = cdsapi.Client()
for year in tqdm(range(1971, 2021)):
    file = f"{folder}/data_{year}.nc"
    if os.path.exists(file):
        print(f"El archivo {file} ya existe")
        continue

    print("Descargando año: ", year)
    for month in range(1, 13):
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "2m_temperature",
                ],
                "year": [f"{year}"],
                "month": [
                    f"{month}",
                ],
                "day": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "27",
                    "28",
                    "29",
                    "30",
                    "31",
                ],
                "time": [
                    "00:00",
                    "01:00",
                    "02:00",
                    "03:00",
                    "04:00",
                    "05:00",
                    "06:00",
                    "07:00",
                    "08:00",
                    "09:00",
                    "10:00",
                    "11:00",
                    "12:00",
                    "13:00",
                    "14:00",
                    "15:00",
                    "16:00",
                    "17:00",
                    "18:00",
                    "19:00",
                    "20:00",
                    "21:00",
                    "22:00",
                    "23:00",
                ],
            },
            file,
        )
        print(f"Se descargó data_{year}.netcdf.zip")
