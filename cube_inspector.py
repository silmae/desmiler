"""
All tools needed to make raw cube into desmiled reflectance cube. 

Assumes the existance of a raw cube, a dark frame, and a frame of reference 
light with sharp spectral lines. White reference is expected to be included 
in the scan.

Usage:

Assuming existance of files test_scan_1_cube.nc, test_scan_1_dark.nc, and test_scan_1_ref.nc. 
Place them in following folder sructure into the same folder where you run the code:

./
    scans/
        test_scan_1/
            test_scan_1_cube.nc
            test_scan_1_dark.nc
            test_scan_1_ref.nc

Then run makeShiftArr(fileName='test_scan_1_ref'), which will create file 
test_scan_1_shift.nc in test_scan_1/ folder.

Run make_all_cubes(test_scan_1_cube), which will create test_scan_1_cube_rfl.nc (reflectance), 
test_scan_1_cube_rfl_intr.nc (desmiled reflectance with interpolative shifts), and test_scan_1_cube_rfl_lut.nc 
(desmiled reflectance with lookup table shifts) in test_scan_1/ folder.

After everything has been run the folder should look like this:

./
    scans/
        test_scan_1/
            test_scan_1_cube.nc
            test_scan_1_cube_rfl.nc
            test_scan_1_cube_rfl_lut.nc
            test_scan_1_cube_rfl_intr.nc
            test_scan_1_dark.nc
            test_scan_1_ref.nc
            test_scan_1_shift.nc

Now you can use CubeShow to inspect the cubes by running:

cs = CubeShow('test_scan_1')
cs.show()

Class 'CubeShow' is an interactive matplotlib-based inspector program 
with simple key and mouse commands.

"""

import numpy as np
import xarray as xr
import os

import smile_correction as smile
import inspector as insp

# Expected dimension names of the spectral cube.
d_along_scan   = 'index'
d_across_scan  = 'y'
d_spectral     = 'x'

cube_data_name = 'dn'

# Along scan white reference area.
white_ref_scan_slice = slice(410,490)


# ------------------------
# Loading stuff from disk.
# ------------------------

def load_cube(scan_name='test_scan_1', cube_type=''):
    """ Loads and returns a cube (xarray Dataset) with given name from ./scans/scan_name/. """
    
    path = f'scans/{scan_name}/{scan_name}_cube{cube_type}.nc'
    print(f"Loading cube {path}...", end=' ')
    ds = xr.open_dataset(os.path.normpath(path))
    return ds

def load_shift_matrix(scan_name='test_scan_1'):
    """ Loads and returns a shift matrix (xarray DataArray) with given name from ./scans/scan_name/. """

    path = f'scans/{scan_name}/{scan_name}_shift.nc'
    da = xr.open_dataarray(os.path.normpath(path))
    return da

def load_dark_frame(scan_name='test_scan_1'):
    """ Loads and returns a dark frame (xarray Dataset) with given name from ./scans/scan_name/. """

    path = f'scans/{scan_name}/{scan_name}_dark.nc'
    ds = xr.open_dataset(os.path.normpath(path))
    return ds.frame

# ------------------------------------------------------------------------
# Manipulating loaded raw data into smile corrected reflectance values.
# ------------------------------------------------------------------------

def make_all_cubes(scan_name='test_scan_1'):
    """ Process and save all cubes on a single run. 
    
    This is a bit faster than calling each step separately because of 
    loading times. 
    """

    rfl = make_reflectance_cube(scan_name=scan_name)
    makeDesmiledCube(scan_name=scan_name, sourceCube=rfl)
    makeDesmiledCubeReIdx(scan_name=scan_name, sourceCube=rfl)

def make_reflectance_cube(scan_name='test_scan_1', sourceCube=None):
    """ Makes a reflectance cube out of given raw cube.
    
    NaNs and Infs that may exist in interpolated desmile cubes are changed 
    into numerical values. Loads the cube if not given. fileName is used 
    for saving the resulting cube, so it must be given even if sourceCube is 
    given.
    
    Resulting cube is saved to disk and returned as xarray Dataset.
    """

    if sourceCube is None:
        org = load_cube(scan_name)
    else:
        org = sourceCube

    print(f"Substracting dark frame...", end=' ')
    d = load_dark_frame(scan_name)
    org['dn_dark_corrected'] = ((d_along_scan, d_across_scan, d_spectral),(org[cube_data_name] > d) * (org[cube_data_name] - d).astype(np.float32))
    org = org.drop(cube_data_name)
    print(f"done")

    print(f"Dividing by white frame...", end=' ')
    # Y coordinates of the reference white (teflon block)
    # Along scan white reference area.
    white_ref_scan_slice = slice(410,490)
    # Along scan white reference area.
    white = (org.dn_dark_corrected.isel({d_along_scan:white_ref_scan_slice})).mean(dim=(d_along_scan)).astype(np.float32)
    rfl = org
    # Uncomment to drop lowest pixel values to zero
    # zeroLessThan = 40
    # rfl = org.where(org.dn_dark_corrected > zeroLessThan, 0.0)
    rfl['reflectance'] = ((d_along_scan, d_across_scan, d_spectral),(rfl.dn_dark_corrected / white).astype(np.float32))
    rfl.reflectance.values = np.nan_to_num(rfl.reflectance.values).astype(np.float32)
    rfl = rfl.drop('dn_dark_corrected')
    print(f"done")

    path = f'scans/{scan_name}/{scan_name}_cube_rfl.nc'
    print(f"Saving reflectance cube to {path}...", end=' ')
    rfl.to_netcdf(os.path.normpath(path), format='NETCDF4', engine='netcdf4')
    print(f"done")


if __name__ == '__main__':
    pass