
import utilities.file_handling as F
import core.properties as P
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt
import analysis.frame_inspector as frame_inspector

base_path = '../../examples/'
example_spectrogram_path = base_path + 'fluorescence_spectrogram.nc'

def light_frame_to_spectrogram():
    """Creates a mean spectrogram from few rows of a frame and saves it.

    Used only for creating an example to be used in creating more complex example data.
    """

    source_session = 'light_test'
    path = '../' + P.path_rel_scan + '/' + source_session + '/' + P.extension_light
    print(f"path: {path}")
    frame_ds = F.load_frame(path)
    frame = frame_ds.frame
    height = frame.y.size
    width = frame.x.size
    half_h = int(height/2)
    crop_hh = 10
    frame = frame.isel({'y':slice(half_h - crop_hh, half_h + crop_hh)})
    frame = frame.mean(dim='y')
    # plt.plot(frame.data)
    # plt.show()
    F.save_frame(frame, example_spectrogram_path)

def make_undistorted_frame():
    """Creates an example of undistorted frame to examples directory.

    Frame data follows closely to the form that camazing uses in the frames it provides.
    Attributes are omitted though.
    """

    perfect_frame_path = os.path.abspath(base_path + 'undistorted_frame.nc')
    if not os.path.exists(perfect_frame_path):
        source = F.load_frame(example_spectrogram_path)
        destination_frame_height = 800
        source_data = source.frame.data
        print(source_data.shape)
        expanded_data = np.repeat(source_data, destination_frame_height)
        expanded_data = np.reshape(expanded_data, (source.frame.x.size, destination_frame_height))
        expanded_data = expanded_data.transpose()
        print(expanded_data.shape)
        coords = {
            "x": ("x", np.arange(0, source.frame.x.size) + 0.5),
            "y": ("y", np.arange(0, destination_frame_height) + 0.5),
            "timestamp": dt.datetime.today().timestamp(),
        }
        dims = ('y', 'x')
        frame = xr.DataArray(
            expanded_data,
            name="frame",
            dims=dims,
            coords=coords,
        )

        #frame_inspector.plot_frame(frame)
        F.save_frame(frame, perfect_frame_path)
    else:
        print(f"Undistorted example frame already exists in '{perfect_frame_path}'. Doing nothing.")


if __name__ == '__main__':
    #light_frame_to_spectrogram()
    make_undistorted_frame()