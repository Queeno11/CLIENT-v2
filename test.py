import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "2m_temperature",
        "year": "1966",
        "month": "02",
        "day": "02",
        "time": "01:00",
    },
    "download.nc",
)
