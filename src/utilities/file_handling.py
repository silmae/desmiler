"""

This file contains file handling for frames, cubes, etc.

"""
import os
import logging
from core import properties as P
from utilities import plotting
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import matplotlib.pyplot as plt

import toml
from toml import TomlDecodeError
import errno


def create_default_directories():
    """Creates default structure if it does not exist yet."""

    print("Checking whether default directories exist. Creating if not.")
    if not os.path.exists(P.path_rel_scan):
        create_directory(P.path_rel_scan)


def create_directory(path:str):
    """Creates a new directory with given relative path string."""

    abs_path = os.path.abspath(path)
    logging.info(f"Creating new directory {abs_path}")
    try:
        os.mkdir(abs_path)
    except OSError:
        logging.warning(f"Creation of the directory {abs_path} failed")
    else:
        logging.info(f"Successfully created the directory {abs_path}")


def save_frame(frame:DataArray, path, meta_dict=None, save_thumbnail=True):
    """Saves a frame to the disk with given name.

    NOTE: even if the frame is expected to be a DataArray object,
    the saved file will be a Dataset object and the name of the
    frame is defined in core.properties.py file as 'naming_frame_data'.

    Creates and saves also a .png image of the frame for easy debugging.

    File extension '.nc' is added if missing.

    Parameters
    ----------
    frame : DataArray
        The frame to be saved on disk.
    path : string or path
        A path to the file.
    meta_dict : Dictionary, optional
        Dictionary of miscellaneous metadata that gets added to DataSet's attributes.
    save_thumbnail : bool, default True
        Whether to save a png of the frame to path along with the actual frame.

    Returns
    -------
    Dataset
        Frame as a dataset in the same format it is saved.
    """

    frame_dataset = xr.Dataset()
    frame_dataset[P.naming_frame_data] = frame

    if meta_dict is not None:
        for key in meta_dict:
            frame_dataset.attrs[key] = meta_dict[key]
    path_s = str(path)

    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'

    abs_path = os.path.abspath(path_s)

    if save_thumbnail:
        thumb_path = os.path.abspath(path_s.rstrip('.nc') + '.png')
        logging.info(f"Saving thumbnail to '{abs_path}'")
        fig,ax = plt.subplots(figsize=plotting.get_figure_size())
        ax.imshow(frame, origin='lower')
        fig.savefig(thumb_path)
        plt.close(fig)

    logging.info(f"Saving frame to '{abs_path}'")
    try:
        frame_dataset.to_netcdf(abs_path, format='NETCDF4')
    finally:
        frame_dataset.close()

    return frame_dataset


def load_frame(path) -> Dataset:
    """Loads a frame from given path.

    Parameters
    ----------
    path : string or path object
        The name of the Dataset to be loaded.
        The '.nc' file ending is added to the name.

    Returns
    -------
    DataSet
        Loaded frame as a Dataset.
    """

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)

    if os.path.exists(abs_path):
        if os.path.isfile(abs_path):
            frame_ds = xr.open_dataset(abs_path)
            frame_ds.close()
            return frame_ds
        else:
            raise RuntimeError(f"Given frame path '{abs_path}' is not a file.")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), abs_path)

def save_cube(cube:Dataset, path):
    """Saves an image cube to given path. """

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)

    cube.to_netcdf(abs_path)
    cube.close()

def load_cube(path):
    """ Loads and returns a cube. Closes file handle once done.

    Returns
    -------
        Dataset
            Loaded cube as xarray Dataset object.

    Raises
    ------
        RuntimeError
            if path is not a file.
        FileNotFoundError
            if path does not exist.
    """

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)

    if os.path.exists(abs_path):
        if os.path.isfile(abs_path):
            cube_ds = xr.open_dataset(abs_path)
            cube_ds.close()
            return cube_ds
        else:
            raise RuntimeError(f"Given cube path '{abs_path}' is not a file.")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), abs_path)


def load_control_file(path):
    """Loads a control file (.toml) from given path.

    File extension .toml added if omitted.

    Returns
    -------
    control
        Contents of the control file as dictionary.
    """

    path_s = str(path)
    if not path_s.endswith('.toml'):
        path_s = path_s + '.toml'
    abs_path = os.path.abspath(path_s)

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

def save_shift_matrix(shift_matrix:DataArray, path):
    """Saves a shift matrix to given path."""

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)

    try:
        shift_matrix.to_netcdf(abs_path)
    finally:
        shift_matrix.close()

def load_shit_matrix(path) -> DataArray:
    """Loads a shift matrix from given path."""

    path_s = str(path)
    if not path_s.endswith('.nc'):
        path_s = path_s + '.nc'
    abs_path = os.path.abspath(path_s)

    shift_matrix = xr.open_dataarray(abs_path)
    shift_matrix.close()
    return shift_matrix

