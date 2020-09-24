
import os
import logging
from core import properties as P

def create_default_directories():
    """Creates default structure if it does not exist yet."""

    print("Checking whether default directories exist. Creating if not.")
    if not os.path.exists(P.scan_folder_name):
        create_directory(P.scan_folder_name)
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