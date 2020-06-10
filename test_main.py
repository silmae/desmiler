import xarray as xr
import os

if __name__ == '__main__':

    test_frame_name = 'test_frame'
    original_frame = xr.open_dataset(os.path.normpath('frames/' + test_frame_name + '.nc')).frame