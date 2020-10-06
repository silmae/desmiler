
import os
import logging
from core import properties as P
import xarray as xr
from xarray import DataArray

import toml
from toml import TomlDecodeError


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


def save_frame(frame:DataArray, path:str, meta_dict=None):
    """Saves a frame to the disk with given name.

    File extension '.nc' is added if missing.

    Parameters
    ----------
    path : string or path
        A path to a folder to which the frame should be saved.
    frame : DataArray
        The frame to be saved on disk.
    meta_dict : Dictionary, optional
        Dictionary of miscellaneous metadata that gets added to DataSet's attributes.
    """

    frameData = xr.Dataset()
    frameData[P.naming_frame_data] = frame

    if meta_dict is not None:
        for key in meta_dict:
            frameData.attrs[key] = meta_dict[key]

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'

    # FIXME This fails if the file is already open.
    frameData.to_netcdf(os.path.normpath(path_s), format='NETCDF4')


def load_frame(path):
    """Loads a frame with given name from the disk.

    Parameters
    ----------
    fileName : string
        The name of the Dataset to be loaded for example 'magic'.
        The '.nc' file ending is added to the name.

    Returns
    -------
    DataArray
        Now, we actually load a Xarray Dataset, but return its 'frame' attribute.
        Don't know if this is a good practise or not.

    Raises
    ------
    FIXME Xarray load exception are left unhandeled. Will fail if the file
    is already opened.
    """

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)
    try:
        frame_ds = xr.open_dataset(abs_path)
        frame_ds.close()
        return frame_ds
    except:
        logging.error(f"Failed to load frame from '{abs_path}'")


def load_control_file(path):

    abs_path = os.path.abspath(path)
    logging.info(f"Searching for existing scan control file from '{abs_path}'")
    if os.path.exists(abs_path):
        print(f"Loading control file")
        try:
            with open(abs_path, 'r') as file:
                scan_settings = toml.load(file)
            print(scan_settings)
            print(f"Control file loaded.")
            return scan_settings
        except TypeError as te:
            print(f"Control file loading failed")
            logging.error(te)
        except TomlDecodeError as tde:
            print(f"Control file loading failed")
            logging.error(tde)
    else:
        logging.warning("Control file not found.")
