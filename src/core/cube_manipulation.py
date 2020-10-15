
import xarray as xr
from xarray import Dataset
from xarray import DataArray
import numpy as np
import logging

import core.properties as P
import core.frame_manipulation as fm

def make_reflectance_cube(raw_cube, dark_frame, white_frame, control) -> Dataset:
    """ Makes a reflectance cube out of a raw cube.

    Raises
    ------
        ValueError
            if white is None
    """


    if white_frame is None:
        # FIXME this is stupid. We can allow option to use an area of the cube as white reference.
        raise ValueError(f"White frame must be provided for reflectance calculations. Was None.")

    rfl = raw_cube.copy(deep=True)
    raw_cube.close()

    if dark_frame is not None:
        df_ds = dark_frame
        df_da = df_ds[P.naming_frame_data]
        dark_frame = fm.crop_to_size(df_da, control)
        print(f"Subtracting dark frame...", end=' ')
        rfl[P.naming_dark_corrected] = (P.dim_order_cube,
                                         (rfl[P.naming_cube_data].values > dark_frame.values)
                                         * (rfl[P.naming_cube_data].values - dark_frame.values).astype(np.float32))
        rfl = rfl.drop(P.naming_cube_data)
        old_data_name = P.naming_dark_corrected
        print(f"done")
    else:
        logging.info(f"Dark frame was not provided for reflectance calculation, so the resulting cube "
                     f"is not corrected for dark current.")
        old_data_name = P.naming_cube_data

    print(f"Dividing by white frame...", end=' ')
    # Y coordinates of the reference white (teflon block)
    # Along scan white reference area.
    use_area_from_cube = False
    if use_area_from_cube:
        white_ref_scan_slice = slice(410, 490)
        # Along scan white reference area.
        # white = (org.dn_dark_corrected.isel({d_along_scan: white_ref_scan_slice})).mean(dim=(d_along_scan)).astype(
        #     np.float32)
    else:
        white = fm.crop_to_size(white_frame, control)
        white = white[P.naming_frame_data]

    # rfl = rfl.transpose(*P.dim_order_cube)
    # # Uncomment to drop lowest pixel values to zero
    # # zeroLessThan = 40
    # # rfl = rfl.where(rfl[P.naming_reflectance] > zeroLessThan, 0.0)
    rfl[P.naming_reflectance] = (P.dim_order_cube, (rfl[old_data_name].values / white.values).astype(np.float32))
    rfl[P.naming_reflectance].values = np.nan_to_num(rfl[P.naming_reflectance].values).astype(np.float32)
    rfl = rfl.drop(old_data_name)
    print(f"done")

    return rfl