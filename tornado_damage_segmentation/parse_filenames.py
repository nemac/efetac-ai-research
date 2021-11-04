# Contains code for altering 'ForWarn_Data' filenames to an acceptable format which can be dealt with by the rest of
# this program. By nature, this is specific to the original filename format used, so I will write a new function here
# for each new format as it comes up. Which function to use can be set in 'config.py'
import os
from datetime import datetime as dt, timedelta
from enum import Enum
from config import raster_dir, filename_format


# Enum of all the parser options
class Parsers(Enum):
    MAXMODIS = 0


# Converts the 'maxMODIS' filenames by changing the date format to %Y-%m-%d
def parse_max_modis_filenames():
    for filename in os.listdir(raster_dir):
        split_filename = filename.split('.')
        year, day = split_filename[1:3]
        # This method from https://stackoverflow.com/questions/2427555/python-question-year-and-day-of-year-to-date
        date = dt(int(year), 1, 1) + timedelta(int(day) - 1)
        split_filename = [split_filename[0]] + [date.isoformat().split('T')[0]] + split_filename[3:]
        os.rename(os.path.join(raster_dir, filename), os.path.join(raster_dir, '.'.join(split_filename)))


# dictionary of parsers to use
parsers = {0: parse_max_modis_filenames}  # TODO: fix

# Runs the selected parser
parsers[filename_format]()
