"""
All tools needed to make raw cube into desmiled reflectance cube. 

Assumes the existance of a raw cube, a dark frame, and a frame of reference 
light with sharp spectral lines. 

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

Run makeAllCubes(test_scan_1_cube), which will create test_scan_1_cube_rfl.nc (reflectance), 
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

import xarray as xr
import os

import smile_correction as smile
import inspector as insp


# ------------------------
# Loading stuff from disk.
# ------------------------

def load_cube(scan_name='test_scan_1', cube_type=''):
    """ Loads and returns a cube (xarray Dataset) with given name from ./scans/scan_name/. """

    path = f'scans/{scan_name}/{scan_name}_cube{cube_type}.nc'
    ds = xr.open_dataset(os.path.normpath(path))
    return ds

def load_shift_matrix(scan_name='test_scan_1'):
    """ Loads and returns a shift matrix (xarray DataArray) with given name from ./data. """

    path = f'scans/{scan_name}/{scan_name}_shift.nc'
    da = xr.open_dataarray(os.path.normpath(path))
    return da

def loadDarkFrame(scan_name='test_scan_1'):
    """ Loads and returns a dark frame (xarray Dataset) with given name from ./frames. """

    path = f'scans/{scan_name}/{scan_name}_dark.nc'
    ds = xr.open_dataset(os.path.normpath(path))
    return ds.frame



if __name__ == '__main__':
