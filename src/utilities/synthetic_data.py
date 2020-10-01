
import utilities.file_handling as F
import core.properties as P
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt
import math
import analysis.frame_inspector as frame_inspector

base_path = '../../examples/'

example_spectrogram_path = os.path.abspath(base_path + 'fluorescence_spectrogram.nc')
undistorted_frame_path = os.path.abspath(base_path + 'undistorted_frame.nc')
smiled_frame_path = os.path.abspath(base_path + 'smiled_frame.nc')

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

    if not os.path.exists(undistorted_frame_path):
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
        F.save_frame(frame, undistorted_frame_path)
    else:
        print(f"Undistorted example frame already exists in '{undistorted_frame_path}'. Doing nothing.")

def load_undistorted_frame():
    if not os.path.exists(undistorted_frame_path):
        make_undistorted_frame()

    frame_ds = F.load_frame(undistorted_frame_path)
    return frame_ds

def shift_matrix_generator(output_array, frame_width):
    """This is the inverse of what would be used to correct a smile effect.

    TODO Should probably move to somplace else.
    """

    circle_center_x = -15000
    circle_center_y = 400
    circle_r = abs(circle_center_x)

    for y in range(len(output_array[:,0])):
        yy = y - circle_center_y
        theta = math.asin(yy / circle_r)
        # Copysign for getting signed distance
        px = (1 - math.cos(theta)) * math.copysign(circle_r, circle_center_x)
        for x in range(frame_width):
            output_array[y, x] = px


def interpolative_shift(frame, distorition_matrix):
    """ Desmile frame using row-wise interpolation of
        pixel intensities.
    """

    distorition_matrix = xr.DataArray(distorition_matrix, dims=('y', 'x'))

    ds = xr.Dataset(
        data_vars={
            'frame': frame,
            'x_shift': distorition_matrix,
        },
    )

    ds['distorted_x'] = ds.x - ds.x_shift
    ds.coords['new_x'] = np.linspace(0, frame.x.size, frame.x.size)
    ds = ds.groupby('y').apply(distort_row)

    ds = ds.drop('x')
    renames = {'new_x': 'x'}
    ds = ds.rename(renames)

    return ds.frame


def distort_row(row):
    """ Used by interpolative shift only. """

    row['x'] = row.distorted_x
    new_x = row.new_x
    row = row.drop(['distorted_x', 'new_x'])
    row = row.interp(x=new_x, method='linear')
    return row

def make_smiled_frame():
    """Creates an example of a frame suffering from spectral smile to examples directory.

    """

    u_frame_ds = load_undistorted_frame()
    u_frame = u_frame_ds.frame
    distortion_array = np.zeros_like(u_frame.data)
    shift_matrix_generator(distortion_array, u_frame.x.size)
    u_frame = interpolative_shift(u_frame, distortion_array)
    F.save_frame(u_frame, smiled_frame_path)
    # plt.imshow(u_frame)
    # plt.show()




if __name__ == '__main__':
    #light_frame_to_spectrogram()
    # make_undistorted_frame()
    #make_smiled_frame()