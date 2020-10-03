
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
distotion_smile_path = base_path + 'distorted' + '_smile'
distotion_tilt_path = base_path + 'distorted' + '_tilt'
distotion_smile_tilt_path = base_path + 'distorted' + '_smile_tilt'

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

    print(f"Generating frame example to '{undistorted_frame_path}'...", end='')
    source = F.load_frame(example_spectrogram_path)
    height = 800
    width = source.frame.x.size
    source_data = source.frame.data
    max_pixel_val = source_data.max()
    print(source_data.shape)
    expanded_data = np.repeat(source_data, height)
    expanded_data = np.reshape(expanded_data, (width, height))
    expanded_data = expanded_data.transpose()
    # print(expanded_data.shape)

    # Multiply each row with a random number
    # rand_row = np.random.uniform(1, 1.10, size=(height,))
    rand_row = np.random.normal(1, 0.03, size=(height,))
    expanded_data = expanded_data * rand_row[:,None]

    # Add random noise
    rando = np.random.uniform(0, 0.05*max_pixel_val, size=(height, width))
    expanded_data = expanded_data + rando

    coords = {
        "x": ("x", np.arange(0, source.frame.x.size) + 0.5),
        "y": ("y", np.arange(0, height) + 0.5),
        "timestamp": dt.datetime.today().timestamp(),
    }
    dims = ('y', 'x')
    frame = xr.DataArray(
        expanded_data,
        name="frame",
        dims=dims,
        coords=coords,
    )

    frame_inspector.plot_frame(frame)
    F.save_frame(frame, undistorted_frame_path)
    print("done")

def load_undistorted_frame():
    if not os.path.exists(undistorted_frame_path):
        make_undistorted_frame()

    frame_ds = F.load_frame(undistorted_frame_path)
    return frame_ds

def generate_distortion_matrix(width, height, method='smile'):
    """This is the inverse of what would be used to correct a smile effect.

    TODO Should probably move to somplace else.
    """

    # height = len(output_array[:, 0])
    # width = len(output_array[0, :])

    distortion_matrix = np.zeros((height,width))

    if method == 'smile':
        circle_center_x = -15000
        circle_center_y = 400
        circle_r = abs(circle_center_x)

        for y in range(height-1):
            yy = y - circle_center_y
            theta = math.asin(yy / circle_r)
            # Copysign for getting signed distance
            px = (1 - math.cos(theta)) * math.copysign(circle_r, circle_center_x)
            for x in range(width-1):
                distortion_matrix[y, x] = px
    if method == 'tilt':
        max_tilt = 10
        col = np.linspace(-int(max_tilt/2),int(max_tilt/2),num=height)
        distortion_matrix = np.repeat(col, width)
        distortion_matrix = np.reshape(distortion_matrix, (height, width))
        # distortion_matrix = distortion_matrix.transpose()

    return distortion_matrix


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

def make_distorted_frame(distortions):
    """Creates an example of a frame suffering from spectral smile to examples directory.

    """

    u_frame_ds = load_undistorted_frame()
    u_frame = u_frame_ds.frame
    save_path = base_path + 'distorted'
    if 'smile' in distortions:
        print(f"adding smile distortion")
        distortion_matrix = generate_distortion_matrix(u_frame.x.size, u_frame.y.size, method='smile')
        u_frame = interpolative_shift(u_frame, distortion_matrix)
        save_path = save_path + '_smile'
    if 'tilt' in distortions:
        print(f"adding tilt distortion")
        distortion_matrix = generate_distortion_matrix(u_frame.x.size, u_frame.y.size, method='tilt')
        u_frame = interpolative_shift(u_frame, distortion_matrix)
        save_path = save_path + '_tilt'

    F.save_frame(u_frame, save_path)
    plt.imshow(u_frame)
    plt.show()

def show_source_spectrogram():
    show_me(example_spectrogram_path)

def show_undistorted_frame():
    show_me(undistorted_frame_path, window_name='Undistorted')

def show_smiled_frame():
    show_me(distotion_smile_path, window_name='Distortions: smile')

def show_tilted_frame():
    show_me(distotion_tilt_path, window_name='Distortions: tilt')

def show_smiled_tilted_frame():
    show_me(distotion_smile_tilt_path, window_name='Distortions: smile + tilt')

def show_me(path, window_name):
    # TODO use frame_inspector instead
    source = F.load_frame(path)
    frame = source.frame
    dim_count = len(frame.dims)
    if dim_count == 1:
        plt.plot(frame.data)
    else:
        frame_inspector.plot_frame(frame, window_name=window_name)

    plt.show()

if __name__ == '__main__':
    # light_frame_to_spectrogram()
    # make_undistorted_frame()
    # make_distorted_frame(['smile'])
    # make_distorted_frame(['tilt'])
    # make_distorted_frame(['smile', 'tilt'])

    # show_source_spectrogram()
    show_undistorted_frame()
    show_smiled_frame()
    show_tilted_frame()
    show_smiled_tilted_frame()
