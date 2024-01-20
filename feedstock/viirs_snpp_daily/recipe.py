import apache_beam as beam
from pangeo_forge_recipes.transforms import (
    StoreToZarr,
    _add_keys,
)
from pangeo_forge_recipes.patterns import ConcatDim
from pangeo_forge_recipes.types import Dimension
import logging
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



class ReadActiveFirePixels(beam.PTransform):

    def __init__(self, columns, renames, fsspec_open_kwargs):
        self.columns = columns
        self.renames = renames
        self.fsspec_open_kwargs = fsspec_open_kwargs

    def read_csv(self, file_pattern: str = None) -> xr.Dataset:
        with fsspec.open(file_pattern, mode='r', **self.fsspec_open_kwargs) as f:
            df = pd.read_csv(
                f,
                parse_dates=[["acq_date", "acq_time"]],
                usecols=self.columns,
                skipinitialspace=True
            )
            df = df.rename(columns=self.renames)
            return xr.Dataset.from_dataframe(df)

    def expand(self, pcoll):
        return (
            pcoll
            | "ReadCSV" >> beam.FlatMap(_add_keys(self.read_csv))
        )


concat_dim = ConcatDim(name="YYYYMMDD_HHMM", keys=list(file_dt_generator()))

viirs_snpp_daily = (
    beam.Create(file_pattern_generator())
    | ReadActiveFirePixels(
        columns=viirs_usecols,
        renames=rename_viirs_columns,
        fsspec_open_kwargs=fsspec_open_kwargs
    )
    | StoreToZarr(
        store_name="viirs_snpp_daily.zarr",
        combine_dims=[Dimension(dim.name, dim.operation) for dim in concat_dim]
    )
)
