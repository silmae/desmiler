
import os
import logging
from core import properties as P
import xarray as xr
from xarray import DataArray

def create_default_directories():
    """Creates default structure if it does not exist yet."""

    print("Checking whether default directories exist. Creating if not.")
    if not os.path.exists(P.path_rel_scan):
        create_directory(P.path_rel_scan)
    if not os.path.exists(P.frame_folder_name):
        create_directory(P.frame_folder_name)

def create_directory(path:str):
    """Creates a new directory with given relative path string."""

    logging.info(f"Creating new directory {path}")
    try:
        os.mkdir(path)
    except OSError:
        logging.warning(f"Creation of the directory {path} failed")
    else:
        logging.info(f"Successfully created the directory {path}")

def save_frame(frame:DataArray, path:str, meta_dict):
    """Saves a frame to the disk with given name.

    File extension '.nc' is added if missing.

    Parameters
    ----------
    path : string or path
        A path to a folder to which the frame should be saved.
    frame : DataArray
        The frame to be saved on disk.
    meta_dict : Dictionary
        Dictionary of miscellaneous metadata that gets added to DataSet's attributes.
    """

    frameData = xr.Dataset()
    frameData[P.naming_frame_data] = frame

    for key in meta_dict:
        frameData.attrs[key] = meta_dict[key]

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'

    # FIXME This fails if the file is already open.
    frameData.to_netcdf(os.path.normpath(path_s), format='NETCDF4')
    