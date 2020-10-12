
import utilities.file_handling as F
import core.properties as P
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray import Dataset
from xarray import DataArray
import datetime as dt
import toml
import math
import logging

import analysis.frame_inspector as frame_inspector
from analysis.cube_inspector import CubeInspector
from core import smile_correction as sc
from imaging.scanning_session import ScanningSession

# TODO rename to 'example_source' and move to top level dir, so that deleting examples folder does not matter?
example_spectrogram_path = os.path.abspath(P.path_example_frames + 'fluorescence_spectrogram.nc')
undistorted_frame_path = os.path.abspath(P.path_example_frames + 'undistorted_frame.nc')
dark_frame_path = os.path.abspath(P.path_example_frames + 'dark.nc')
distotion_smile_path = P.path_example_frames + 'distorted' + '_smile'
distotion_tilt_path = P.path_example_frames + 'distorted' + '_tilt'
distotion_smile_tilt_path = P.path_example_frames + 'distorted' + '_smile_tilt'
shift_path = P.path_example_frames + 'shift.nc'
desmile_lut_path = P.path_example_frames + 'desmiled_lut.nc'
desmile_intr_path = P.path_example_frames + 'desmiled_intr.nc'

# Height of the (fictive) example sensor.
sensor_height = 2704
# Height of the effective area of the sensor that the slit can illuminate
slit_height = 800

random_noise_fac = 0.07
row_noise_fac = 0.03

cube_depth = 160
stripe_width = 40

default_tilt = -1.0
default_curvature = -30e-6

key_curvature = 'generated_curvature'
key_tilt = 'generated_tilt'

def light_frame_to_spectrogram():
    """Creates a mean spectrogram from few rows of a frame and saves it.

    Used only for creating an example to be used in creating more complex example data.
    """

    source_session = 'light_test'
    path = '../' + P.path_rel_scan + '/' + source_session + '/' + P.ref_light_name
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
    expanded_data = np.repeat(source_data, height)
    expanded_data = np.reshape(expanded_data, (width, height))
    expanded_data = expanded_data.transpose()
    # print(expanded_data.shape)

    full_sensor = np.zeros((sensor_height, width))
    fh2 = int(sensor_height / 2)
    sh2 = int(slit_height/2)
    full_sensor[fh2-sh2:fh2+sh2,:] = expanded_data

    # Multiply each row with a random number
    # rand_row = np.random.uniform(1, 1.10, size=(frame_height,))
    rand_row = np.random.normal(1, row_noise_fac, size=(sensor_height,))
    full_sensor = full_sensor * rand_row[:,None]

    # Add random noise
    rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=(sensor_height, width))
    full_sensor = full_sensor + rando

    coords = {
        "x": ("x", np.arange(0, source.frame.x.size) + 0.5),
        "y": ("y", np.arange(0, sensor_height) + 0.5),
        "timestamp": dt.datetime.today().timestamp(),
    }
    dims = ('y', 'x')
    frame = xr.DataArray(
        full_sensor,
        name="frame",
        dims=dims,
        coords=coords,
    )

    # frame_inspector.plot_frame(frame)
    F.save_frame(frame, undistorted_frame_path)
    print("done")

def make_dark_frame():
    """Creates an example of dark frame (just noise) to examples directory.

    Frame data follows closely to the form that camazing uses in the frames it provides.
    Attributes are omitted though.
    """

    print(f"Generating dark frame example to '{dark_frame_path}'...", end='')
    source = F.load_frame(example_spectrogram_path)
    width = source.frame.x.size
    source_data = source.frame.data
    max_pixel_val = source_data.max()

    full_sensor = np.zeros((sensor_height, width))

    # Add random noise
    rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=(sensor_height, width))
    full_sensor = full_sensor + rando

    coords = {
        "x": ("x", np.arange(0, source.frame.x.size) + 0.5),
        "y": ("y", np.arange(0, sensor_height) + 0.5),
        "timestamp": dt.datetime.today().timestamp(),
    }
    dims = ('y', 'x')
    frame = xr.DataArray(
        full_sensor,
        name="frame",
        dims=dims,
        coords=coords,
    )

    # frame_inspector.plot_frame(frame)
    F.save_frame(frame, dark_frame_path)
    print("done")

def load_undistorted_frame():
    if not os.path.exists(undistorted_frame_path):
        make_undistorted_frame()

    frame_ds = F.load_frame(undistorted_frame_path)
    return frame_ds

def generate_distortion_matrix(width, height, amount, method='smile') -> DataArray:
    """Generates a distortion matrix.

    This is the inverse of what would be used to correct a smile effect.

    Parameters
    ----------
        width: int
            Width of the generated matrix
        height: int
            Height of the generated matrix
        amount: float
            Amount of tilt in degrees if method is 'tilt' or
            amount of curvature if method is 'smile'. Negative tilt value tilts spectral
            lines to the right and positive to the left. Negative curvature causes curvature
            opening to the right and positive to the left.
        method: str
            Distortion method. Either 'tilt' or 'smile'.
    """

    distortion_matrix = np.zeros((height,width))

    if method == 'smile':
        curvature = amount
        circle_r = 1 / curvature
        circle_center_x = ((width / 2) + circle_r)
        circle_center_y = int(height/2)

        for y in range(height-1):
            yy = y - circle_center_y
            theta = math.asin(yy / circle_r)
            # Copysign for getting signed distance
            px = (1 - math.cos(theta)) * math.copysign(circle_r, circle_center_x)
            for x in range(width-1):
                distortion_matrix[y, x] = px
    if method == 'tilt':
        tilt_deg = amount
        tilt_px = math.sin(math.radians(tilt_deg)) * height
        col = np.linspace(-int(tilt_px/2),int(tilt_px/2),num=height)
        distortion_matrix = np.repeat(col, width)
        distortion_matrix = np.reshape(distortion_matrix, (height, width))

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

def make_distorted_frame(distortions, amount=None):
    """Creates an example of a frame suffering from spectral smile or tilt or both to examples directory."""

    print("Generating distorted frame")
    u_frame_ds = load_undistorted_frame()
    u_frame = u_frame_ds.frame
    width = u_frame.x.size
    height = u_frame.y.size
    save_path = P.path_example_frames + 'distorted'
    meta = {}

    if 'smile' in distortions:
        if amount is None:
            curvature = default_curvature
        else:
            curvature = amount
        meta[key_curvature] = curvature
        distortion_matrix = generate_distortion_matrix(width, height, curvature, method='smile')
        u_frame = interpolative_shift(u_frame, distortion_matrix)
        save_path = save_path + '_smile'
    if 'tilt' in distortions:
        if amount is None:
            tilt = default_tilt
        else:
            tilt = amount
        meta[key_tilt] = tilt
        distortion_matrix = generate_distortion_matrix(width, height, tilt, method='tilt')
        save_path = save_path + '_tilt'
        u_frame = interpolative_shift(u_frame, distortion_matrix)

    F.save_frame(u_frame, save_path, meta)
    print(f"Generated distorted frame to '{save_path}'")
    # plt.imshow(u_frame)
    # plt.show()

def make_stripe_cube():

    print(f"Generating stripe example raw cube.")

    control = toml.loads(P.example_scan_control_content)
    width = control['scan_settings']['width']
    width_offset = control['scan_settings']['width_offset']
    height = control['scan_settings']['height']
    height_offset = control['scan_settings']['height_offset']

    if not os.path.exists(distotion_smile_tilt_path + '.nc'):
        make_distorted_frame(['smile', 'tilt'])

    white_area = F.load_frame(distotion_smile_tilt_path)
    dark_area = white_area.copy(deep=True)

    F.save_frame(white_area.frame, P.path_rel_scan + P.example_scan_name + '/' + P.ref_white_name)
    F.save_frame(dark_area.frame, P.path_rel_scan + P.example_scan_name + '/' + P.ref_dark_name)
    shift = F.load_shit_matrix(shift_path)
    F.save_shift_matrix(shift, P.path_rel_scan + P.example_scan_name + '/' + P.shift_name)

    x_slice = slice(width_offset, width_offset + width)
    y_slice = slice(height_offset, height_offset + height)
    white_area = white_area.isel({P.dim_x: x_slice,P.dim_y: y_slice})
    dark_area = dark_area.isel({P.dim_x: x_slice, P.dim_y: y_slice})
    white_area.frame.values = np.nan_to_num(white_area.frame.values)
    white_area['x'] = np.arange(0, white_area.x.size) + 0.5
    white_area['y'] = np.arange(0, white_area.y.size) + 0.5
    dark_area['x'] = np.arange(0, white_area.x.size) + 0.5
    dark_area['y'] = np.arange(0, white_area.y.size) + 0.5
    area_shape = white_area.frame.values.shape
    max_pixel_val = white_area.frame.max().item()
    dark_area.frame.values = np.random.uniform(0, random_noise_fac*max_pixel_val,size=area_shape)

    frame_list = []
    stripe_counter = 0
    use_white = True
    for i in range(cube_depth):
        if stripe_counter > stripe_width-1:
            use_white = not use_white
            stripe_counter = 0
        rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=area_shape)
        if use_white:
            f = white_area.copy(deep=True)
            f.frame.values = f.frame.values + rando
        else:
            f = dark_area.copy(deep=True)
            f.frame.values = rando

        f.coords[P.dim_scan] = i
        frame_list.append(f.frame)
        stripe_counter += 1

    frames = xr.concat(frame_list, dim=P.dim_scan)
    cube = xr.Dataset(
        data_vars={
            'dn': frames,
        },
    )
    F.save_cube(cube, P.path_rel_scan + '/' + P.example_scan_name + '/' + P.cube_raw_name)
    print(f"Generated stripe example raw cube.")

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
    # frame_inspector.plot_frame(light_frame, sl_list, True, True, False, 'testing')

    abs_path = os.path.abspath(shift_path)
    print(f"Saving shift matrix to {abs_path}...", end=' ')
    shift_matrix.to_netcdf(abs_path)
    print("done")
    # Uncomment for debugging
    # shift_matrix.plot.imshow()
    # plt.show()
    return shift_matrix

def apply_frame_correction(shift_matrix, method):
    control = toml.loads(P.example_scan_control_content)
    width = control['scan_settings']['width']
    width_offset = control['scan_settings']['width_offset']
    height = control['scan_settings']['height']
    height_offset = control['scan_settings']['height_offset']

    positions = np.array(control['spectral_lines']['positions']) - width_offset
    peak_width = control['spectral_lines']['peak_width']
    bandpass_width = control['spectral_lines']['window_width']

    light_ds = F.load_frame(distotion_smile_tilt_path)
    light_ds = light_ds.isel({'x': slice(width_offset, width_offset + width),
                              'y': slice(height_offset, height_offset + height)})

    light_frame = light_ds.frame
    corrected = sc.apply_shift_matrix(light_frame, shift_matrix=shift_matrix, method=method, target_is_cube=False)
    # Uncomment for debugging
    # corrected.plot.imshow()
    # plt.show()
    return corrected

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

def show_desmiled_lut():
    show_me(desmile_lut_path, window_name='Desmiled with LUT')

def show_desmiled_intr():
    show_me(desmile_intr_path, window_name='Desmiled with INTR')

def show_me(path, window_name=None):
    source = F.load_frame(path)
    frame = source.frame
    dim_count = len(frame.dims)
    if dim_count == 1:
        plt.plot(frame.data)
    else:
        frame_inspector.plot_frame(frame, window_name=window_name)

    plt.show()

def generate_cube_examples():
    check_dirs()
    make_stripe_cube()
    session = ScanningSession(P.example_scan_name)
    session.make_reflectance_cube()
    session.desmile_cube(shift_method=0)
    session.desmile_cube(shift_method=1)

def generate_frame_examples():
    check_dirs()
    make_undistorted_frame()
    make_dark_frame()
    make_distorted_frame(['smile'])
    make_distorted_frame(['tilt'])
    make_distorted_frame(['smile', 'tilt'])

    sm = make_shift_matrix()

    lut_frame = apply_frame_correction(sm, 0)
    F.save_frame(lut_frame, desmile_lut_path)
    intr_frame = apply_frame_correction(sm, 1)
    F.save_frame(intr_frame, desmile_intr_path)

def generate_all_examples():
    check_dirs()
    generate_frame_examples()
    generate_cube_examples()

def check_dirs():
    F.create_default_directories()
    if not os.path.exists(P.path_rel_scan + P.example_scan_name):
        F.create_directory(P.path_rel_scan + P.example_scan_name)
    if not os.path.exists(P.path_example_frames):
        F.create_directory(P.path_example_frames)

def show_frame_examples():
    show_source_spectrogram()
    show_undistorted_frame()
    show_smiled_frame()
    show_tilted_frame()
    show_smiled_tilted_frame()

def show_shift_matrix():
    sm = F.load_shit_matrix(shift_path)
    sm.plot.imshow()
    plt.show()

def show_cube_examples():
    try:
        rfl = F.load_cube(P.path_rel_scan + P.example_scan_name + '/' + P.cube_reflectance_name)
        desmiled_lut = F.load_cube(P.path_rel_scan + P.example_scan_name + '/' + P.cube_desmiled_lut)
        desmiled_intr = F.load_cube(P.path_rel_scan + P.example_scan_name + '/' + P.cube_desmiled_intr)
        ci = CubeInspector(rfl, desmiled_lut, desmiled_intr, 'reflectance')
        ci.show()
    except FileNotFoundError as fnf:
        logging.error(fnf)
        print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")
    except RuntimeError as r:
        logging.error(r)
        print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")

def show_raw_cube():
    try:
        raw = F.load_cube(P.path_rel_scan + P.example_scan_name + '/' + P.cube_raw_name)
        ci = CubeInspector(raw, raw, raw, P.naming_cube_data)
        ci.show()
    except FileNotFoundError as fnf:
        logging.error(fnf)
        print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")
    except RuntimeError as r:
        logging.error(r)
        print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")

if __name__ == '__main__':
    # light_frame_to_spectrogram()

    # generate_frame_examples()
    # generate_cube_examples()

    # show_frame_examples()

    show_raw_cube()
    show_cube_examples()
    # show_shift_matrix()

