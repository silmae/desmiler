
import utilities.file_handling as F
import core.properties as P
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt
import toml
import math
import analysis.frame_inspector as frame_inspector
from core import smile_correction as sc

base_path = '../../examples/'

example_spectrogram_path = os.path.abspath(base_path + 'fluorescence_spectrogram.nc')
undistorted_frame_path = os.path.abspath(base_path + 'undistorted_frame.nc')
distotion_smile_path = base_path + 'distorted' + '_smile'
distotion_tilt_path = base_path + 'distorted' + '_tilt'
distotion_smile_tilt_path = base_path + 'distorted' + '_smile_tilt'
shift_path = base_path + 'shift.nc'

# Height of the sensor
frame_height = 2704
# Height of the effective area that the slit can illuminate
slit_height = 800

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
    height = slit_height
    width = source.frame.x.size
    source_data = source.frame.data
    max_pixel_val = source_data.max()
    print(source_data.shape)
    expanded_data = np.repeat(source_data, height)
    expanded_data = np.reshape(expanded_data, (width, height))
    expanded_data = expanded_data.transpose()
    # print(expanded_data.shape)

    full_sensor = np.zeros((frame_height, width))
    fh2 = int(frame_height/2)
    sh2 = int(slit_height/2)
    full_sensor[fh2-sh2:fh2+sh2,:] = expanded_data

    # Multiply each row with a random number
    # rand_row = np.random.uniform(1, 1.10, size=(frame_height,))
    rand_row = np.random.normal(1, 0.03, size=(frame_height,))
    full_sensor = full_sensor * rand_row[:,None]

    # Add random noise
    rando = np.random.uniform(0, 0.05*max_pixel_val, size=(frame_height, width))
    full_sensor = full_sensor + rando

    coords = {
        "x": ("x", np.arange(0, source.frame.x.size) + 0.5),
        "y": ("y", np.arange(0, frame_height) + 0.5),
        "timestamp": dt.datetime.today().timestamp(),
    }
    dims = ('y', 'x')
    frame = xr.DataArray(
        full_sensor,
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

    TODO Should probably move to someplace else.
    """

    # height = len(output_array[:, 0])
    # width = len(output_array[0, :])

    distortion_matrix = np.zeros((height,width))

    if method == 'smile':
        curvature = 30e-6
        circle_r = 1 / curvature
        # 1 curves left and -1 curves right
        curving_direction = -1
        circle_center_x = ((width / 2) + circle_r) * curving_direction
        circle_center_y = int(height/2)

        for y in range(height-1):
            yy = y - circle_center_y
            theta = math.asin(yy / circle_r)
            # Copysign for getting signed distance
            px = (1 - math.cos(theta)) * math.copysign(circle_r, circle_center_x)
            for x in range(width-1):
                distortion_matrix[y, x] = px
    if method == 'tilt':
        tilt_deg = -1
        max_tilt = math.sin(math.radians(tilt_deg)) * height
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
    # plt.imshow(u_frame)
    # plt.show()


def make_shift_matrix():
    """Make shift matrix and save it to disk.

    If there does not exist a file './scan_name/scan_name_shift.nc', this method
    has to be called to create one. The shift matrix is valid for all cubes
    imaged with same settings (hardware and software).
    """

    control = toml.loads(P.example_scan_control_content)
    width = control['scan_settings']['width']
    width_offset = control['scan_settings']['width_offset']
    height = control['scan_settings']['height']
    height_offset = control['scan_settings']['height_offset']

    positions = np.array(control['spectral_lines']['positions']) - width_offset
    peak_width = control['spectral_lines']['peak_width']
    bandpass_width = control['spectral_lines']['window_width']


    load_path = distotion_smile_tilt_path
    light_ds = F.load_frame(load_path)
    light_ds = light_ds.isel({'x':slice(width_offset, width_offset + width),
                              'y':slice(height_offset, height_offset + height)})
    light_frame = light_ds.frame
    bp = sc.construct_bandpass_filter(light_frame, positions, bandpass_width)
    sl_list = sc.construct_spectral_lines(light_frame, positions, bp)
    shift_matrix = sc.construct_shift_matrix(sl_list, light_frame.x.size, light_frame.y.size)
    frame_inspector.plot_frame(light_frame, sl_list, True, True, False, 'testing')

    print(f"Saving shift matrix to {shift_path}...", end=' ')
    shift_matrix.to_netcdf(os.path.normpath(shift_path))
    print("done")
    # Uncomment for debugging
    shift_matrix.plot.imshow()
    plt.show()
    return shift_matrix

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

def show_me(path, window_name=None):
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
    # show_undistorted_frame()
    # show_smiled_frame()
    # show_tilted_frame()
    # show_smiled_tilted_frame()

    make_shift_matrix()
