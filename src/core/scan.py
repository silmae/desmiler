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

Then run make_shift_matrix(fileName='test_scan_1'), which will create file 
test_scan_1_shift.nc in test_scan_1/ folder.

Run make_all_cubes(test_scan_1), which will create test_scan_1_cube_rfl.nc (reflectance), 
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

Now you can use CubeInspector to inspect the cubes by running:

ci = CubeInspector('test_scan_1')
ci.show()

"""

import numpy as np
import xarray as xr
import os

from core import smile_correction as smile

# Expected dimension names of the spectral cube.
d_along_scan   = 'index'
d_across_scan  = 'y'
d_spectral     = 'x'

# Under what name the data is in the cube. 
cube_data_name = 'dn'

# Along scan white reference area.
white_ref_scan_slice = slice(410,490)


# ------------------------
# Loading stuff from disk.
# ------------------------

def load_cube(scan_name='test_scan_1', cube_type=None):
    """ Loads and returns a cube (xarray Dataset) with given name from ./scans/scan_name/. """
    
    if cube_type == None:
        ct = ''
    else:
        ct = f'_{cube_type}'
    path = f'scans/{scan_name}/{scan_name}_cube{ct}.nc'
    print(f"Loading cube {path}...", end=' ')
    ds = xr.open_dataset(os.path.normpath(path))
    print('done')
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
    desmile_cube(scan_name=scan_name, source_cube=rfl, shift_method=0)
    desmile_cube(scan_name=scan_name, source_cube=rfl, shift_method=1)

def make_reflectance_cube(scan_name='test_scan_1', source_cube=None):
    """ Makes a reflectance cube out of a raw cube.
    
    Loads the cube if not given. 
    
    Resulting cube is saved into scans/{scan_name}/{scan_name}_cube_rfl.nc and 
    returned as xarray Dataset.    
    """

    if source_cube is None:
        org = load_cube(scan_name)
    else:
        org = source_cube

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

def desmile_cube(scan_name='test_scan_1', source_cube=None, shift_method=0):
    """ Desmile a reflectance cube with lut of intr shifts and save and return the result."""

    if shift_method == 0:
        cube_type = 'lut'
    elif shift_method == 1:
        cube_type = 'intr'

    if source_cube is None:
        path = f'scans/{scan_name}/{scan_name}_cube_rfl.nc'
        rfl = load_cube(scan_name, 'rfl')
    else:
        rfl = source_cube

    s = load_shift_matrix(scan_name)

    print(f"Desmiling {cube_type} shifts...", end=' ')
    desmiled = smile.apply_shift_matrix(rfl, s, method=shift_method, target_is_cube=True)
    print(f"done")

    path = f'scans/{scan_name}/{scan_name}_cube_rfl_{cube_type}.nc'
    print(f"Saving desmiled cube to {path}...", end=' ')
    desmiled.to_netcdf(os.path.normpath(path))
    print(f"done")
    return desmiled

def make_shift_matrix(locations, bandpass_width=30, scan_name='test_scan_1'):
    """Make shift matrix and save it to disk.
    
    If there does not exist a file './scan_name/scan_name_shift.nc', this method 
    has to be called to create one. The shift matrix is valid for all cubes 
    imaged with same settings (hardware and software).
    """

    load_path = f'scans/{scan_name}/{scan_name}_ref.nc'
    ds = xr.load_dataset(os.path.normpath(load_path))
    frame = ds.frame
    bp = smile.construct_bandpass_filter(frame, locations, bandpass_width)
    sl_list = smile.construct_spectral_lines(frame, locations, bp)
    shift_matrix = smile.construct_shift_matrix(sl_list, frame[d_spectral].size,  frame[d_across_scan].size)

    save_path = f'scans/{scan_name}/{scan_name}_shift.nc'
    print(f"Saving shift matrix to {save_path}...", end=' ')
    shift_matrix.to_netcdf(os.path.normpath(save_path))
    print("done")
    # Uncomment for debugging
    # shift_matrix.plot.imshow()
    return shift_matrix

if __name__ == '__main__':

    # Tester code. After running these, the cubes can be inspeced with CubeShow.
    # Hard coded locations of some spectral lines. 
    locations = [360,430,515,665,780,827,930,970,1025,1382,2060]
    make_shift_matrix(locations, bandpass_width=30, scan_name='test_scan_1')
    make_all_cubes(scan_name='test_scan_1')