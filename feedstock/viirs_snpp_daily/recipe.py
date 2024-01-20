import uuid

import apache_beam as beam
from pangeo_forge_recipes.transforms import (
    StoreToZarr,
    _add_keys,
)
from pangeo_forge_recipes.types import Dimension, CombineOp
from pangeo_forge_recipes.patterns import FilePattern, ConcatDim
import logging
from typing import List, Dict
from datetime import datetime, timedelta
logger = logging.getLogger('apache-beam')
import xarray as xr
import fsspec
import pandas as pd
import os

# import xarray as xr
# import fsspec
# import pandas as pd
# columns = [
#     "latitude",
#     "longitude",
#     "daynight",
#     "frp",
#     #"confidence",
#     #"scan",
#     #"track",
#     "acq_date",
#     "acq_time",
# ]
# renames = {
#     "latitude": "Lat",
#     "longitude": "Lon",
#     "frp": "FRP",
#     #"scan": "DS",
#     #"track": "DT",
#     "acq_date_acq_time": "YYYYMMDD_HHMM",
# }
# path='s3://gcorradini-forge-runner-test/snpp_daily/SUOMI_VIIRS_C2_Global_VNP14IMGTDL_NRT_2023240.txt'
# with fsspec.open(path, mode='r') as f:
#     df = pd.read_csv(
#         f,
#         parse_dates=[["acq_date", "acq_time"]],
#         usecols=columns,
#         skipinitialspace=True
#     )
#     df = df.rename(columns=renames)
#     # to add indexes for xarray dimensions we need to reduce the dtypes
#     df = df.astype( {
#         'Lat': 'float32',
#         'Lon': 'float32',
#         'FRP': 'float16',
#     })
#     df = df.set_index(['YYYYMMDD_HHMM'])
#     ds = xr.Dataset.from_dataframe(df)
#     # ds = ds.set_coords('Lat')
#     # ds = ds.set_coords('Lon')
#     # ds = ds.set_coords('YYYYMMDD_HHMM')



viirs_usecols = [
    "latitude",
    "longitude",
    #"daynight",
    "frp",
    #"confidence",
    #"scan",
    #"track",
    "acq_date",
    "acq_time",
]

rename_viirs_columns = {
    "latitude": "Lat",
    "longitude": "Lon",
    "frp": "FRP",
    #"scan": "DS",
    #"track": "DT",
    "acq_date_acq_time": "YYYYMMDD_HHMM",
}

fsspec_open_kwargs = {
    'key': os.environ["S3_DEFAULT_AWS_ACCESS_KEY_ID"],
    'secret': os.environ["S3_DEFAULT_AWS_SECRET_ACCESS_KEY"],
    "anon": False,
    'client_kwargs': {
        'region_name': 'us-west-2'
    }
}


def file_dt_generator(begin=(2023, 9, 9), end=(2023, 9, 16)):
    begin_dt, end_dt = datetime(*begin), datetime(*end)
    while begin_dt <= end_dt:
        yield begin_dt.strftime("%Y%j")
        begin_dt += timedelta(days=1)


def file_pattern_generator(YYYYMMDD_HHMM):
    return f's3://gcorradini-forge-runner-test/snpp_daily/SUOMI_VIIRS_C2_Global_VNP14IMGTDL_NRT_{YYYYMMDD_HHMM}.txt'


pattern = FilePattern(
    file_pattern_generator,
    ConcatDim(name="YYYYMMDD_HHMM", keys=list(file_dt_generator())),
)


def read_csv(file_path: str, columns: List[str], renames: Dict, fsspec_open_kwargs: Dict) -> xr.Dataset:
    with fsspec.open(file_path, mode='r', **fsspec_open_kwargs) as f:
        df = pd.read_csv(
            f,
            parse_dates=[["acq_date", "acq_time"]],
            usecols=columns,
            skipinitialspace=True
        )
        df = df.rename(columns=renames)
        # add index so xr.DataSet converts to dimensions
        df = df.set_index(['YYYYMMDD_HHMM'])
        ds = xr.Dataset.from_dataframe(df)
        #ds = ds.chunk(1000)
        return ds


class ReadActiveFirePixels(beam.PTransform):

    def __init__(self, columns, renames, fsspec_open_kwargs):
        self.columns = columns
        self.renames = renames
        self.fsspec_open_kwargs = fsspec_open_kwargs

    def expand(self, pcoll):
        return pcoll | "ReadCSV" >> beam.Map(
                _add_keys(read_csv),
                self.columns,
                self.renames,
                fsspec_open_kwargs
            )


viirs_snpp_daily = (
    beam.Create(pattern.items())
    | ReadActiveFirePixels(
        columns=viirs_usecols,
        renames=rename_viirs_columns,
        fsspec_open_kwargs=fsspec_open_kwargs
    )
    | StoreToZarr(
        store_name="viirs_snpp_daily.zarr",
        combine_dims=pattern.combine_dim_keys,
        target_chunks={'YYYYMMDD_HHMM': 1000}
    )
)
