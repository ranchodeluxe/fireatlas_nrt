import uuid

import apache_beam as beam
from pangeo_forge_recipes.transforms import (
    StoreToZarr,
    _add_keys,
)
from pangeo_forge_recipes.types import Dimension, CombineOp
import logging
from typing import List, Dict
from datetime import datetime, timedelta
logger = logging.getLogger('apache-beam')
import xarray as xr
import fsspec
import pandas as pd
import os


viirs_usecols = [
    "latitude",
    "longitude",
    "daynight",
    "frp",
    "confidence",
    "scan",
    "track",
    "acq_date",
    "acq_time",
]

rename_viirs_columns = {
    "latitude": "Lat",
    "longitude": "Lon",
    "frp": "FRP",
    "scan": "DS",
    "track": "DT",
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


def file_dt_generator(begin=(2023, 9, 1), end=(2023, 9, 30)):
    begin_dt, end_dt = datetime(*begin), datetime(*end)
    while begin_dt <= end_dt:
        yield begin_dt.strftime("%Y%j")
        begin_dt += timedelta(days=1)


def file_pattern_generator():
    for dt_str in file_dt_generator():
        yield f's3://gcorradini-forge-runner-test/snpp_daily/SUOMI_VIIRS_C2_Global_VNP14IMGTDL_NRT_{dt_str}.txt'


def read_csv(file_pattern:str, columns: List[str], renames: Dict, fsspec_open_kwargs: Dict) -> xr.Dataset:
    with fsspec.open(file_pattern, mode='r', **fsspec_open_kwargs) as f:
        df = pd.read_csv(
            f,
            parse_dates=[["acq_date", "acq_time"]],
            usecols=columns,
            skipinitialspace=True
        )
        df = df.rename(columns=renames)
        return (str(uuid.uuid4())[:8], xr.Dataset.from_dataframe(df))


class ReadActiveFirePixels(beam.PTransform):

    def __init__(self, columns, renames, fsspec_open_kwargs):
        self.columns = columns
        self.renames = renames
        self.fsspec_open_kwargs = fsspec_open_kwargs

    def expand(self, pcoll):
        return pcoll | "ReadCSV" >> beam.Map(
                _add_keys(read_csv),
                columns=self.columns,
                renames=self.renames,
                fsspec_open_kwargs=fsspec_open_kwargs
            )


viirs_snpp_daily = (
    beam.Create(file_pattern_generator())
    | ReadActiveFirePixels(
        columns=viirs_usecols,
        renames=rename_viirs_columns,
        fsspec_open_kwargs=fsspec_open_kwargs
    )
    | StoreToZarr(
        store_name="viirs_snpp_daily.zarr",
        combine_dims=[Dimension("YYYYMMDD_HHMM", CombineOp.CONCAT),]
    )
)