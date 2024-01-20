import apache_beam as beam
from pangeo_forge_recipes.patterns import FilePattern, ConcatDim
import logging
from typing import List, Dict
from datetime import datetime, timedelta
logger = logging.getLogger('apache-beam')
import fsspec
import pandas as pd
import os

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

output_prefix = 's3://gcorradini-forge-runner-test/snpp_daily/'

fsspec_open_kwargs = {
    'key': os.environ["S3_DEFAULT_AWS_ACCESS_KEY_ID"],
    'secret': os.environ["S3_DEFAULT_AWS_SECRET_ACCESS_KEY"],
    "anon": False,
    'client_kwargs': {
        'region_name': 'us-west-2'
    }
}


def file_dt_generator(begin=(2023, 11, 21), end=(2024, 1, 20)):
    begin_dt, end_dt = datetime(*begin), datetime(*end)
    while begin_dt <= end_dt:
        yield begin_dt.strftime("%Y%j")
        begin_dt += timedelta(days=1)


def file_pattern_generator(dt_str):
    return f'https://nrt4.modaps.eosdis.nasa.gov/api/v2/content/archives/FIRMS/suomi-npp-viirs-c2/Global/SUOMI_VIIRS_C2_Global_VNP14IMGTDL_NRT_{dt_str}.txt'


pattern = FilePattern(
    file_pattern_generator,
    ConcatDim(name="dt_str", keys=list(file_dt_generator())),
)


def download_csv(input_file_path: str, columns: List[str], renames: Dict, fsspec_open_kwargs: Dict, output_path_prefix: str) -> str:
    file_name = os.path.basename(input_file_path)

    with fsspec.open(input_file_path, mode='r') as f:
        df = pd.read_csv(
            f,
            parse_dates=[["acq_date", "acq_time"]],
            usecols=columns,
            skipinitialspace=True
        )
        df = df.rename(columns=renames)

    output_file_path = os.path.join(output_path_prefix, file_name)
    with fsspec.open(output_file_path, mode='w', **fsspec_open_kwargs) as of:
        df.to_csv(of)
    return output_file_path



class DownloadViirsData(beam.PTransform):

    def __init__(self, columns, renames, fsspec_open_kwargs, output_prefix):
        self.columns = columns
        self.renames = renames
        self.fsspec_open_kwargs = fsspec_open_kwargs
        self.output_prefix = output_prefix

    def expand(self, pcoll):
        return pcoll | "DownloadCSV" >> beam.Map(
                download_csv,
                self.columns,
                self.renames,
                self.fsspec_open_kwargs,
                self.output_prefix
            )


viirs_data = (
    beam.Create(pattern.items())
    | DownloadViirsData(
        columns=viirs_usecols,
        renames=rename_viirs_columns,
        fsspec_open_kwargs=fsspec_open_kwargs,
        output_prefix=output_prefix
    )
)
