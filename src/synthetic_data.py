"""

This file contains synthetic data generation and inspection. This is only for examples and experiments
and does not affect the usage of imaging.

In your IDE, you can set this file to be run (working directory must be set to the same directory
where this file lives) and run different methods to play around with the example data.

"""

import utilities.file_handling as F
import core.properties as P
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt
import toml
import math
import logging

import analysis.frame_inspector as frame_inspector
from analysis.cube_inspector import CubeInspector
from core import smile_correction as sc
from imaging.scanning_session import ScanningSession

# Define paths used to various source and destination files that are created for examples.
example_spectrogram_path = os.path.abspath(P.path_example_frames + 'fluorescence_spectrogram.nc')
undistorted_frame_path = os.path.abspath(P.path_example_frames + 'undistorted_frame.nc')
dark_frame_path = os.path.abspath(P.path_example_frames + 'dark.nc')
distortion_smile_path = P.path_example_frames + 'distorted' + '_smile'
distortion_tilt_path = P.path_example_frames + 'distorted' + '_tilt'
distortion_smile_tilt_path = P.path_example_frames + 'distorted' + '_smile_tilt'
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
default_curvature = -3e-5

# TODO remove these and use the keys in properties file
key_curvature_generated = 'generated_curvature'
key_tilt_generated = 'generated_tilt'
key_curvature_measured_mean = 'measured_mean_curvature'
key_tilt_measured_mean = 'measured_mean_tilt'

#
# This is not to be used unless the data in version control needs to be changed.
#
# DO NOT REMOVE THIS METHOD
#
# def light_frame_to_spectrogram():
#     """Creates a mean spectrogram from few rows of a frame and saves it.
#
#     Used only for creating 'fluorescence_spectrogram.nc' example spectrogram, which is included
#     in version control. Usage requires recorded real data!
#     """
#
#     source_session = 'light_test'
#     path = '../' + P.path_rel_scan + '/' + source_session + '/' + P.ref_light_name
#     print(f"path: {path}")
#     frame_ds = F.load_frame(path)
#     frame = frame_ds[P.naming_frame_data]
#     height = frame[P.dim_y].size
#     width = frame[P.dim_x].size
#     half_h = int(height / 2)
#     crop_hh = 10
#     frame = frame.isel({P.dim_y: slice(half_h - crop_hh, half_h + crop_hh)})
#     frame = frame.mean(dim=P.dim_y)
#     # plt.plot(frame.data)
#     # plt.show()
#     F.save_frame(frame, example_spectrogram_path)

def make_undistorted_and_dark_frame():
    """Creates an example of undistorted frame and dark frame to examples directory.

    Created frame is "full sensor size" where the area illuminated by the slit
    is centered vertically. Use global variables 'row_noise_fac' and 'random_noise_fac'
    to control the level of added random noise.

    Frame data follows closely to the form that camazing uses in the frames it provides.
    Attributes are omitted though.
    """

    print(f"Generating frame example to '{undistorted_frame_path}'...", end='')
    source = F.load_frame(example_spectrogram_path)
    height = slit_height
    width = source[P.naming_frame_data][P.dim_x].size
    source_data = source[P.naming_frame_data].data
    max_pixel_val = source_data.max()
    expanded_data = np.repeat(source_data, height)
    expanded_data = np.reshape(expanded_data, (width, height))
    expanded_data = expanded_data.transpose()

    full_sensor = np.zeros((sensor_height, width))
    fh2 = int(sensor_height / 2)
    sh2 = int(slit_height / 2)
    full_sensor[fh2 - sh2:fh2 + sh2, :] = expanded_data

    # Multiply each row with a random number
    rand_row = np.random.normal(1, row_noise_fac, size=(sensor_height,))
    full_sensor = full_sensor * rand_row[:, None]

    # Add random noise
    rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=(sensor_height, width))
    full_sensor = full_sensor + rando

    coords = {
        P.dim_x: (P.dim_x, np.arange(0, source[P.naming_frame_data][P.dim_x].size) + 0.5),
        P.dim_y: (P.dim_y, np.arange(0, sensor_height) + 0.5),
        "timestamp": dt.datetime.today().timestamp(),
    }
    dims = (P.dim_y, P.dim_x)
    frame = xr.DataArray(
        full_sensor,
        name=P.naming_frame_data,
        dims=dims,
        coords=coords,
    )

    # frame_inspector.plot_frame(frame)
    F.save_frame(frame, undistorted_frame_path)
    print("done")

    print(f"Generating dark frame example to '{dark_frame_path}'...", end='')
    dark = frame.copy(deep=True)
    dark.data = rando
    dark_frame = xr.DataArray(
        dark,
        name=P.naming_frame_data,
        dims=dims,
        coords=coords,
    )
    F.save_frame(dark_frame, dark_frame_path)
    print("done")

def generate_distortion_matrix(width, height, amount, method='smile') -> np.ndarray:
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

    distortion_matrix = np.zeros((height, width))

    if method == 'smile':
        curvature = amount
        circle_r = 1 / curvature
        circle_center_x = ((width / 2) + circle_r)
        circle_center_y = int(height / 2)

        for y in range(height - 1):
            yy = y - circle_center_y
            theta = math.asin(yy / circle_r)
            # Copysign for getting signed distance
            px = (1 - math.cos(theta)) * math.copysign(circle_r, circle_center_x)
            for x in range(width - 1):
                distortion_matrix[y, x] = px
    if method == 'tilt':
        tilt_deg = amount
        tilt_px = math.sin(math.radians(tilt_deg)) * height
        col = np.linspace(-int(tilt_px / 2), int(tilt_px / 2), num=height)
        distortion_matrix = np.repeat(col, width)
        distortion_matrix = np.reshape(distortion_matrix, (height, width))

    return distortion_matrix

def interpolative_distortion(frame, distorition_matrix):
    """ Use interpolation to apply the distortion matrix to undistorted frame.

    This does cause NaNs in the resulting frame.
    """

    distorition_matrix = xr.DataArray(distorition_matrix, dims=(P.dim_y, P.dim_x))

    ds = xr.Dataset(
        data_vars={
            P.naming_frame_data: frame,
            'x_shift': distorition_matrix,
        },
    )

    ds['distorted_x'] = ds[P.dim_x] - ds.x_shift
    ds.coords['new_x'] = np.linspace(0, frame[P.dim_x].size, frame[P.dim_x].size)
    ds = ds.groupby(P.dim_y).apply(distort_row)

    ds = ds.drop(P.dim_x)
    renames = {'new_x': P.dim_x}
    ds = ds.rename(renames)

    return ds[P.naming_frame_data]

def distort_row(row):
    """ Row-wise interpolation. """

    row[P.dim_x] = row.distorted_x
    new_x = row.new_x
    row = row.drop(['distorted_x', 'new_x'])
    row = row.interp({P.dim_x: new_x}, method='linear')
    return row

def make_distorted_frame(distortions, amount=None):
    """Creates an example of a frame suffering from spectral smile or tilt or both to examples directory.

    Adds metadata to the frame, which can be accessed through the dataset ds by 'ds.attributes'.
    Meta contains tilt and curvature of the spectral lines.
    """

    print("Generating distorted frame")
    if not os.path.exists(undistorted_frame_path):
        make_undistorted_and_dark_frame()

    u_frame_ds = F.load_frame(undistorted_frame_path)
    u_frame = u_frame_ds[P.naming_frame_data]
    width = u_frame[P.dim_x].size
    height = u_frame[P.dim_y].size
    save_path = P.path_example_frames + 'distorted'
    meta = {}

    if 'smile' in distortions:
        if amount is None:
            curvature = default_curvature
        else:
            curvature = amount
        meta[key_curvature_generated] = curvature
        distortion_matrix = generate_distortion_matrix(width, height, curvature, method='smile')
        u_frame = interpolative_distortion(u_frame, distortion_matrix)
        save_path = save_path + '_smile'
    if 'tilt' in distortions:
        if amount is None:
            tilt = default_tilt
        else:
            tilt = amount
        meta[key_tilt_generated] = tilt
        distortion_matrix = generate_distortion_matrix(width, height, tilt, method='tilt')
        save_path = save_path + '_tilt'
        u_frame = interpolative_distortion(u_frame, distortion_matrix)

    ################
    # Find spectral lines and add the mean values to metadata to verify correctness of
    # the calculated values.
    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]
    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    peak_width = control[P.ctrl_spectral_lines][P.ctrl_peak_width]
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    crop_frame = u_frame.isel({P.dim_x: slice(width_offset, width_offset + width),
                               P.dim_y: slice(height_offset, height_offset + height)})
    bp = sc.construct_bandpass_filter(crop_frame, positions, bandpass_width)
    sl_list = sc.construct_spectral_lines(crop_frame, positions, bp, peak_width=peak_width)

    meta[P.meta_key_sl_count] = len(sl_list)
    meta[P.meta_key_location] = [sl.location for sl in sl_list]
    meta[P.meta_key_tilt] = [sl.tilt for sl in sl_list]
    meta[P.meta_key_curvature] = [sl.curvature for sl in sl_list]

    meta[key_curvature_measured_mean] = np.mean(np.array([sl.curvature for sl in sl_list]))
    meta[key_tilt_measured_mean] = np.mean(np.array([sl.tilt_angle_degree_abs for sl in sl_list]))

    print(meta)

    ################

    u_frame = u_frame.isel({P.dim_x: slice(width_offset, width_offset + width),
                              P.dim_y: slice(height_offset, height_offset + height)})

    F.save_frame(u_frame, save_path, meta)
    print(f"Generated distorted frame to '{save_path}'")
    # Uncomment for debugging
    # plt.imshow(u_frame)
    # plt.show()

def make_stripe_cube():
    """Generates a raw cube.

    The cube is as if black and white target illuminated with fluorescence
    light was scanned. The result is saved to example_scan directory along
    with generated dark and white frames (copied from frame_examples directory),
    which will later be needed to calculate reflectance images. Also shift
    matrix is copied from frame_examples.
    """

    print(f"Generating stripe example raw cube.")

    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

    if not os.path.exists(distortion_smile_tilt_path + '.nc'):
        make_distorted_frame(['smile', 'tilt'])

    white_area = F.load_frame(distortion_smile_tilt_path)
    dark_area = white_area.copy(deep=True)

    F.save_frame(white_area[P.naming_frame_data], P.path_rel_scan + P.example_scan_name + '/' + P.ref_white_name)
    F.save_frame(dark_area[P.naming_frame_data], P.path_rel_scan + P.example_scan_name + '/' + P.ref_dark_name)
    shift = F.load_shit_matrix(shift_path)
    F.save_shift_matrix(shift, P.path_rel_scan + P.example_scan_name + '/' + P.shift_name)

    x_slice = slice(width_offset, width_offset + width)
    y_slice = slice(height_offset, height_offset + height)
    white_area = white_area.isel({P.dim_x: x_slice, P.dim_y: y_slice})
    dark_area = dark_area.isel({P.dim_x: x_slice, P.dim_y: y_slice})
    white_area[P.naming_frame_data].values = np.nan_to_num(white_area[P.naming_frame_data].values)
    white_area[P.dim_x] = np.arange(0, white_area[P.dim_x].size) + 0.5
    white_area[P.dim_y] = np.arange(0, white_area[P.dim_y].size) + 0.5
    dark_area[P.dim_x] = np.arange(0, white_area[P.dim_x].size) + 0.5
    dark_area[P.dim_y] = np.arange(0, white_area[P.dim_y].size) + 0.5
    area_shape = white_area[P.naming_frame_data].values.shape
    max_pixel_val = white_area[P.naming_frame_data].max().item()
    dark_area[P.naming_frame_data].values = np.random.uniform(0, random_noise_fac * max_pixel_val, size=area_shape)

    frame_list = []
    stripe_counter = 0
    use_white = True
    for i in range(cube_depth):
        if stripe_counter > stripe_width - 1:
            use_white = not use_white
            stripe_counter = 0
        rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=area_shape)
        if use_white:
            f = white_area.copy(deep=True)
            f[P.naming_frame_data].values = f[P.naming_frame_data].values + rando
        else:
            f = dark_area.copy(deep=True)
            f[P.naming_frame_data].values = rando

        f.coords[P.dim_scan] = i
        frame_list.append(f[P.naming_frame_data])
        stripe_counter += 1

    frames = xr.concat(frame_list, dim=P.dim_scan)
    cube = xr.Dataset(
        data_vars={
            P.naming_cube_data: frames,
        },
    )
    F.save_cube(cube, P.path_rel_scan + '/' + P.example_scan_name + '/' + P.cube_raw_name)
    print(f"Generated stripe example raw cube.")

def make_shift_matrix():
    """Make shift matrix and save it to disk."""

    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    light_ds = F.load_frame(distortion_smile_tilt_path)

    light_ds = light_ds.isel({P.dim_x: slice(width_offset, width_offset + width),
                              P.dim_y: slice(height_offset, height_offset + height)})
    light_frame = light_ds[P.naming_frame_data]
    bp = sc.construct_bandpass_filter(light_frame, positions, bandpass_width)
    sl_list = sc.construct_spectral_lines(light_frame, positions, bp)
    shift_matrix = sc.construct_shift_matrix(sl_list, light_frame[P.dim_x].size, light_frame[P.dim_y].size)
    # frame_inspector.plot_frame(light_frame, sl_list, True, True, False, 'testing')

    abs_path = os.path.abspath(shift_path)
    print(f"Saving shift matrix to {abs_path}...", end=' ')
    shift_matrix.to_netcdf(abs_path)
    print("done")
    # Uncomment for debugging
    # shift_matrix.plot.imshow()
    # plt.show()
    return shift_matrix, sl_list

def apply_frame_correction(method):
    """Do a smile correction for a single frame and return the result.

    Parameters
    ----------
        method: int
            Either 0 for lookup table method or 1 for row interpolation method.
    """
    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

    light_ds = F.load_frame(distortion_smile_tilt_path)
    light_ds = light_ds.isel({P.dim_x: slice(width_offset, width_offset + width),
                              P.dim_y: slice(height_offset, height_offset + height)})
    sm, sl = make_shift_matrix()

    # Uncomment for debugging
    # frame_inspector.plot_frame(light_ds, sl, True, True)
    # sm.plot()
    # plt.show()

    light_frame = light_ds[P.naming_frame_data]
    corrected = sc.apply_shift_matrix(light_frame, shift_matrix=sm, method=method, target_is_cube=False)
    return corrected

def show_source_spectrogram():
    """Show the spectrogram used to generate all the examples. """

    _show_a_frame(example_spectrogram_path)

def show_undistorted_frame():
    """Show generated ideal frame free of distortions."""

    _show_a_frame(undistorted_frame_path, window_name='Undistorted')

def show_dark_frame():
    """Show generated dark frame."""

    _show_a_frame(dark_frame_path, window_name='Dark current')

def show_smiled_frame():
    """Show generated frame with just smile distortion."""

    _show_a_frame(distortion_smile_path, window_name='Distortions: smile')

def show_tilted_frame():
    """Show generated frame with just tilt distortion."""

    _show_a_frame(distortion_tilt_path, window_name='Distortions: tilt')

def show_smiled_tilted_frame():
    """Show generated frame with smile and tilt distortion.

    This resembles most closely the real frames acquired from the imager.
    """

    _show_a_frame(distortion_smile_tilt_path, window_name='Distortions: smile + tilt')

def show_desmiled_lut():
    """Show the smile and tilt distorted frame after running smile correction with lookup table shifts."""

    _show_a_frame(desmile_lut_path, window_name='Desmiled with LUT')

def show_desmiled_intr():
    """Show the smile and tilt distorted frame after running smile correction with interpolated shifts."""

    _show_a_frame(desmile_intr_path, window_name='Desmiled with INTR')

def _show_a_frame(path, window_name=None):
    """General method to show various stuff. """

    source = F.load_frame(path)
    frame = source[P.naming_frame_data]
    dim_count = len(frame.dims)
    if dim_count == 1:
        plt.plot(frame.data)
    else:
        frame_inspector.plot_frame(source, window_name=window_name)

    plt.show()

def generate_cube_examples():
    """Generate cube examples. This takes some time and RAM."""

    check_dirs()
    make_stripe_cube()
    session = ScanningSession(P.example_scan_name)
    session.make_reflectance_cube()
    session.desmile_cube(shift_method=0)
    session.desmile_cube(shift_method=1)

def generate_frame_examples():
    """Generate all of the frame examples. Fairly fast operation."""

    check_dirs()
    make_undistorted_and_dark_frame()
    make_distorted_frame(['smile'])
    make_distorted_frame(['tilt'])
    make_distorted_frame(['smile', 'tilt'])

    lut_frame = apply_frame_correction(0)
    F.save_frame(lut_frame, desmile_lut_path)
    intr_frame = apply_frame_correction(1)
    F.save_frame(intr_frame, desmile_intr_path)

def generate_all_examples():
    """Generates all the cube and frame examples. Takes a lot of time and memory."""

    check_dirs()
    generate_frame_examples()
    generate_cube_examples()

def check_dirs():
    """Check that necessary directories exist."""

    F.create_default_directories()
    if not os.path.exists(P.path_rel_scan + P.example_scan_name):
        F.create_directory(P.path_rel_scan + P.example_scan_name)
    if not os.path.exists(P.path_example_frames):
        F.create_directory(P.path_example_frames)

def show_frame_examples():
    """Shows all frame examples one by one. """

    show_source_spectrogram()
    show_dark_frame()
    show_undistorted_frame()
    show_smiled_frame()
    show_tilted_frame()
    show_smiled_tilted_frame()

def show_shift_matrix():
    """Show the example shift matrix."""

    sm = F.load_shit_matrix(shift_path)
    sm.plot.imshow()
    plt.show()

def show_cube_examples():
    """Show cube examples using CubeInspector. """

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
    """Show only the raw cube with the CubeInspector.

    This is mainly for debugging convenience so that the desmiling does not have to be run to test the CubeInspector.
    """
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

    generate_frame_examples()
    # generate_cube_examples()
    # generate_all_examples()

    show_frame_examples()

    # show_raw_cube()
    # show_cube_examples()
    # show_smiled_tilted_frame()

    # apply_frame_correction(0)
    # apply_frame_correction(1)
    # show_desmiled_intr()
    # show_desmiled_lut()
