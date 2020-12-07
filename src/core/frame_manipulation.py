"""

Unfinished frame manipulation file. Should contain all common frame operations.

"""
import numpy as np
import xarray as xr

import core.properties as P

def crop_to_size(source_frame, control):
    """Crops a frame as dictated by given control dictionary.

    Resets the coordinates as if the frame was acquired with cropping dimensions,
    i.e., the coordinates will start from 0.5 and run to frame.dim.size - 0.5.

    Parameters
    ----------
        source_frame: xarray Dataset
            Frame to be cropped.
        control: dict
            Control file content as a dictionary.
    """
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]
    frame = source_frame.isel({P.dim_x: slice(width_offset, width_offset + width),
                        P.dim_y: slice(height_offset, height_offset + height)})

    frame[P.dim_x] = np.arange(0, frame.x.size) + 0.5
    frame[P.dim_y] = np.arange(0, frame.y.size) + 0.5
    # Ensure that the the dimensions are in the right order.
    frame = frame.transpose(*P.dim_order_frame)
    return frame
