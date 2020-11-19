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

import time

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

# FIXME remove these and use the keys in properties file
key_curvature_generated = 'generated_curvature'
key_tilt_generated = 'generated_tilt'
key_curvature_measured_mean = 'measured_mean_curvature'
key_tilt_measured_mean = 'measured_mean_tilt'


def light_frame_to_spectrogram():
    """Creates a mean spectrogram from few rows of a frame and saves it.

    Used only for creating 'fluorescence_spectrogram.nc' example spectrogram, which is included
    in version control. Usage requires recorded real data!
    """

    source_session = 'light_test'
    path = '../' + P.path_rel_scan + '/' + source_session + '/' + P.ref_light_name
    print(f"path: {path}")
    frame_ds = F.load_frame(path)
    frame = frame_ds[P.naming_frame_data]
    height = frame[P.dim_y].size
    width = frame[P.dim_x].size
    half_h = int(height / 2)
    crop_hh = 10
    frame = frame.isel({P.dim_y: slice(half_h - crop_hh, half_h + crop_hh)})
    frame = frame.mean(dim=P.dim_y)
    # plt.plot(frame.data)
    # plt.show()
    F.save_frame(frame, example_spectrogram_path)


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


# def make_dark_frame():
#     """Creates an example of dark frame (just noise) to examples directory.
#
#     Frame data follows closely to the form that camazing uses in the frames it provides.
#     Attributes are omitted though.
#     """
#
#
#     source = F.load_frame(example_spectrogram_path)
#     width = source[P.naming_frame_data][P.dim_x].size
#     source_data = source[P.naming_frame_data].data
#     max_pixel_val = source_data.max()
#
#     full_sensor = np.zeros((sensor_height, width))
#
#     # Add random noise
#     rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=(sensor_height, width))
#     full_sensor = full_sensor + rando
#
#     coords = {
#         "x": ("x", np.arange(0, source[P.naming_frame_data][P.dim_x].size) + 0.5),
#         "y": ("y", np.arange(0, sensor_height) + 0.5),
#         "timestamp": dt.datetime.today().timestamp(),
#     }
#     dims = ('y', 'x')
#     frame = xr.DataArray(
#         full_sensor,
#         name="frame",
#         dims=dims,
#         coords=coords,
#     )
#
#     # frame_inspector.plot_frame(frame)
#     F.save_frame(frame, dark_frame_path)
#     print("done")


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

    Adds metadata to the frame. Meta contains tilt and curvature of the spectral lines.
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
        plt.imshow(distortion_matrix)
        plt.show()
        u_frame = interpolative_distortion(u_frame, distortion_matrix)
        save_path = save_path + '_smile'
    if 'tilt' in distortions:
        if amount is None:
            tilt = default_tilt
        else:
            tilt = amount
        meta[key_tilt_generated] = tilt
        distortion_matrix = generate_distortion_matrix(width, height, tilt, method='tilt')
        plt.imshow(distortion_matrix)
        plt.show()
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

    # TODO the mean curvature is not very good estimator as shallow curves may be in both directions
    meta[key_curvature_measured_mean] = np.mean(np.array([sl.curvature for sl in sl_list]))
    meta[key_tilt_measured_mean] = np.mean(np.array([sl.tilt_angle_degree_abs for sl in sl_list]))

    print(meta)

    ################

    u_frame = u_frame.isel({P.dim_x: slice(width_offset, width_offset + width),
                              P.dim_y: slice(height_offset, height_offset + height)})

    F.save_frame(u_frame, save_path, meta)
    print(f"Generated distorted frame to '{save_path}'")
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

    if not os.path.exists(distotion_smile_tilt_path + '.nc'):
        make_distorted_frame(['smile', 'tilt'])

    white_area = F.load_frame(distotion_smile_tilt_path)
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
    """Make shift matrix and save it to disk.

    TODO move elsewhere
    """

    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    peak_width = control[P.ctrl_spectral_lines][P.ctrl_peak_width]
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    light_ds = F.load_frame(distotion_smile_tilt_path)

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
    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    peak_width = control[P.ctrl_spectral_lines][P.ctrl_peak_width]
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    light_ds = F.load_frame(distotion_smile_tilt_path)
    light_ds = light_ds.isel({P.dim_x: slice(width_offset, width_offset + width),
                              P.dim_y: slice(height_offset, height_offset + height)})
    sm, sl = make_shift_matrix()

    # Uncomment for debugging
    frame_inspector.plot_frame(light_ds, sl, True, True)
    sm.plot()
    plt.show()

    light_frame = light_ds[P.naming_frame_data]
    corrected = sc.apply_shift_matrix(light_frame, shift_matrix=sm, method=method, target_is_cube=False)
    return corrected


def show_source_spectrogram():
    show_me(example_spectrogram_path)


def show_undistorted_frame():
    show_me(undistorted_frame_path, window_name='Undistorted')


def show_dark_frame():
    show_me(dark_frame_path, window_name='Dark current')


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
    frame = source[P.naming_frame_data]
    dim_count = len(frame.dims)
    if dim_count == 1:
        plt.plot(frame.data)
    else:
        frame_inspector.plot_frame(source, window_name=window_name)

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
    make_undistorted_and_dark_frame()
    # make_dark_frame()
    make_distorted_frame(['smile'])
    make_distorted_frame(['tilt'])
    make_distorted_frame(['smile', 'tilt'])

    lut_frame = apply_frame_correction(0)
    F.save_frame(lut_frame, desmile_lut_path)
    intr_frame = apply_frame_correction(1)
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
    show_dark_frame()
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

def generate_frame_series(frame_count=10):
    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]
    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    peak_width = control[P.ctrl_spectral_lines][P.ctrl_peak_width]
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    source = F.load_frame(example_spectrogram_path)
    source_data = source[P.naming_frame_data].data
    source_data = source_data[width_offset:width_offset + width]

    np.random.seed(3226546)

    def frame_gen():

        max_pixel_val = source_data.max()
        expanded_data = np.repeat(source_data, height)
        expanded_data = np.reshape(expanded_data, (width, height))
        expanded_data = expanded_data.transpose()

        # Multiply each row with a random number
        rand_row = np.random.normal(1, row_noise_fac, size=(height,))
        expanded_data = expanded_data * rand_row[:, None]

        # Add random noise
        rando = np.random.uniform(0, random_noise_fac * max_pixel_val, size=(height, width))
        expanded_data = expanded_data + rando

        coords = {
            P.dim_x: (P.dim_x, np.arange(0, width) + 0.5),
            P.dim_y: (P.dim_y, np.arange(0, height) + 0.5),
            "timestamp": dt.datetime.today().timestamp(),
        }
        dims = (P.dim_y, P.dim_x)
        frame = xr.DataArray(
            expanded_data,
            name=P.naming_frame_data,
            dims=dims,
            coords=coords,
        )
        return frame

    print("Generating distorted frame series")

    save_path = P.path_example_frames + 'distorted_series'

    curvature = -3e-5
    tilt = -1
    use_intr = True

    smile_matrix = generate_distortion_matrix(width, height, amount=curvature, method='smile')
    tilt_matrix = generate_distortion_matrix(width, height, amount=tilt, method='tilt')
    distorition_matrix = xr.DataArray(smile_matrix + tilt_matrix, dims=(P.dim_y, P.dim_x))

    meta = {}
    meta[key_curvature_generated] = curvature
    meta[key_tilt_generated] = tilt

    frame_list = [None]*frame_count
    print(f"Starting frame generator loop for {frame_count} frames.")
    frames_generated = 0
    time_loop_start = time.perf_counter()
    for i in range(frame_count):
        percent = frames_generated / frame_count
        dur = (time.perf_counter() - time_loop_start) / 60
        if frames_generated > 0:
            eta = (dur / frames_generated) * (frame_count - frames_generated)
            print(f"{percent*100:.0f} % ({frames_generated}) of the frames were generated in "
                  f"{dur:.2f} minutes, ETA in {eta:.0f} minutes")

        u_frame = frame_gen()
        for key in meta:
            u_frame.attrs[key] = meta[key]

        if use_intr:
            ds = xr.Dataset(
                data_vars={
                    P.naming_frame_data: u_frame,
                    'x_shift': distorition_matrix,
                },
            )

            ds['distorted_x'] = ds[P.dim_x] - ds.x_shift
            ds.coords['new_x'] = np.linspace(0, u_frame[P.dim_x].size, u_frame[P.dim_x].size)
            ds = ds.groupby(P.dim_y).apply(distort_row)

            ds = ds.drop(P.dim_x)
            renames = {'new_x': P.dim_x}
            ds = ds.rename(renames)

            frame = ds[P.naming_frame_data]
            frame.coords[P.dim_scan] = i
            frame_list[i] = frame
            frames_generated += 1
        else:
            raise NotImplementedError(f"LUT distortion not implemented")
    print(f"Loop finished. ")

    print(f"Saving frame series as a cube.")
    frames = xr.concat(frame_list, dim=P.dim_scan)
    cube = xr.Dataset(
        data_vars={
            P.naming_cube_data: frames,
        },
    )
    F.save_cube(cube, save_path)
    print(f"Series cube saved to {save_path}")

def show_distorted_series():
    """Debugging method for series. """

    save_path = P.path_example_frames + 'distorted_series'
    distorted_series = F.load_cube(save_path)
    frame_count = distorted_series[P.naming_cube_data][P.dim_scan].size
    for i in range(frame_count):
        print(f"Showing frame {i}")
        frame = distorted_series[P.naming_cube_data].isel({P.dim_scan:i})

        if len(frame.attrs) >= 1:
            print(f"Frame metadata from DataArray:")
            for key, val in frame.attrs.items():
                print(f"\t{key} : \t{val}")

        plt.imshow(frame)
        plt.show()

def show_corrected_series():
    """Debugging method for series. """

    save_path = P.path_example_frames + 'corrected_series'
    distorted_series = F.load_cube(save_path)
    frame_count = distorted_series[P.naming_cube_data][P.dim_scan].size
    for i in range(frame_count):
        print(f"Showing frame {i}")
        frame = distorted_series[P.naming_cube_data].isel({P.dim_scan:i})
        frame_attrs = distorted_series['frame_attrs'].isel({P.dim_scan:i})

        if len(frame.attrs) >= 1:
            print(f"Frame metadata from DataArray:")
            for key, val in frame.attrs.items():
                print(f"\t{key} : \t{val}")

        for i in range(frame_attrs.attr_idx.size):
            attr_name = frame_attrs.coords['attr_idx'].values[i]
            attr_vals = frame_attrs.isel({'attr_idx':i}).values
            mean = np.mean(frame_attrs.isel({'attr_idx':i}).values)
            std = np.std(frame_attrs.isel({'attr_idx':i}).values)
            print(f"{attr_name} : {attr_vals} -- {mean} pm {std}")


        # plt.imshow(frame)
        # plt.show()

    sl_count = distorted_series['sl_idx'].size
    for i in range(sl_count):
        sl = distorted_series['frame_attrs'].isel({'sl_idx':i})
        means = sl.mean(dim={'scan_index'})

        print(f"mean tilt for sl {i}:\t {means[1].values} -> {means[4].values}")
    for i in range(sl_count):
        sl = distorted_series['frame_attrs'].isel({'sl_idx': i})
        means = sl.mean(dim={'scan_index'})
        print(f"mean curvature for sl {i}: {means[2].values} -> {means[5].values}")

def desmile_series(first=0, last=1000):
    """Desmile and save metadata of the result """

    print(f"Starting series correction from {first} to {last}.")

    flag_fail = False

    control = toml.loads(P.example_scan_control_content)
    width = control[P.ctrl_scan_settings][P.ctrl_width]
    width_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
    height = control[P.ctrl_scan_settings][P.ctrl_height]
    height_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]
    positions = np.array(control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
    peak_width = control[P.ctrl_spectral_lines][P.ctrl_peak_width]
    bandpass_width = control[P.ctrl_spectral_lines][P.ctrl_window_width]

    save_path = P.path_example_frames + 'corrected_series'
    load_path = P.path_example_frames + 'distorted_series'
    distorted_series = F.load_cube(load_path)
    distorted_series = distorted_series.isel({P.dim_scan:slice(first,last+1)})
    frame_count = distorted_series[P.naming_cube_data][P.dim_scan].size

    frame_list = []
    attr_list = []

    attr_names = [P.meta_key_location, P.meta_key_tilt, P.meta_key_curvature,
                  P.meta_key_location + '_c', P.meta_key_tilt + '_c', P.meta_key_curvature + '_c']
    attr_count = len(attr_names)

    def save_part(last_success):
        if last_success >= first:
            save_path_part = os.path.abspath(save_path + f'_{first}_{last_success}')
            print(f"Saving frame series from {first} to {last_success} as a cube.")
            frames = xr.concat(frame_list, dim=P.dim_scan)
            frame_attrs = xr.concat(attr_list, dim=P.dim_scan)
            cube = xr.Dataset(
                data_vars={
                    P.naming_cube_data: frames,
                    'frame_attrs': frame_attrs,
                },
            )
            F.save_cube(cube, save_path_part)
            print(f"Series cube saved to {save_path_part}")

        next_frame = last_success + 2
        if next_frame < last:
            print(f"Restarting desmiling from frame {next_frame}")
            desmile_series(next_frame, last)

    frames_generated = 0
    curr_frame = first
    time_loop_start = time.perf_counter()
    for _ in range(first, last+1):

        percent = frames_generated / frame_count
        dur = (time.perf_counter() - time_loop_start) / 60
        if frames_generated > 0:
            eta = (dur / frames_generated) * (frame_count - frames_generated)
            print(f"{percent * 100:.0f} % ({frames_generated}/{frame_count}) of the frames were corrected in "
                  f"{dur:.2f} minutes, ETA in {eta:.0f} minutes")
        # try:
        frame = distorted_series[P.naming_cube_data].isel({P.dim_scan:curr_frame-first})
        # frame = distorted_series[P.naming_cube_data]
        # except:
        #     print(f"Problem in isel")
        #     break

        bp = sc.construct_bandpass_filter(frame, positions, bandpass_width)
        sl_list = sc.construct_spectral_lines(frame, positions, bp, peak_width=peak_width)
        sl_count = len(sl_list)

        if sl_count != 4:
            print(f"Frame {curr_frame} failed.")
            break

        # Add metadata for uncorrected spectral lines
        attr_shape = (len(sl_list), attr_count)
        attr_matrix = np.zeros(attr_shape, np.float)
        # frame.attrs[P.meta_key_sl_count]  = len(sl_list)
        attr_matrix[:, 0] = [sl.location for sl in sl_list]
        attr_matrix[:, 1] = [sl.tilt for sl in sl_list]
        attr_matrix[:, 2] = [sl.curvature for sl in sl_list]

        shift_matrix = sc.construct_shift_matrix(sl_list, frame[P.dim_x].size, frame[P.dim_y].size)
        frame = sc.apply_shift_matrix(frame, shift_matrix=shift_matrix, method=0, target_is_cube=False)
        sl_list_corrected = sc.construct_spectral_lines(frame, positions, bp, peak_width=peak_width)
        
        if len(sl_list_corrected) != 4:
            print(f"Frame {curr_frame}, not enough lines found in corrected frame.")
            break

        attr_matrix[:, 3] = [sl.location for sl in sl_list_corrected]
        attr_matrix[:, 4] = [sl.tilt for sl in sl_list_corrected]
        attr_matrix[:, 5] = [sl.curvature for sl in sl_list_corrected]

        frame.coords[P.dim_scan] = curr_frame
        frame_list.append(frame)
        # attr_da = xr.DataArray(attr_matrix)

        coords = {
            'sl_idx': ('sl_idx', np.arange(0, len(sl_list))),
            'attr_idx': ('attr_idx', attr_names),
        }
        dims = ('sl_idx', 'attr_idx')
        attr_da = xr.DataArray(
            attr_matrix,
            name='attr_array',
            dims=dims,
            coords=coords,
        )

        attr_da.coords[P.dim_scan] = curr_frame
        attr_list.append(attr_da)
        frames_generated += 1
        curr_frame += 1

    save_part(curr_frame-1)

#     # TODO the mean curvature is not very good estimator as shallow curves may be in both directions
#     meta[key_curvature_measured_mean] = np.mean(np.array([sl.curvature for sl in sl_list]))
#     meta[key_tilt_measured_mean] = np.mean(np.array([sl.tilt_angle_degree_abs for sl in sl_list]))
#
#     print(meta)
#
#     ################
#
#     u_frame = u_frame.isel({P.dim_x: slice(width_offset, width_offset + width),
#                               P.dim_y: slice(height_offset, height_offset + height)})
#
#     F.save_frame(u_frame, save_path, meta)
#     print(f"Generated distorted frame to '{save_path}'")
#     # plt.imshow(u_frame)
#     # plt.show()

if __name__ == '__main__':

    # generate_frame_series(1000)
    # show_distorted_series()
    desmile_series(0,999)
    # show_corrected_series()

    # light_frame_to_spectrogram()

    # generate_frame_examples()
    # generate_cube_examples()
    # generate_all_examples()

    # show_frame_examples()

    # show_raw_cube()
    # show_cube_examples()
    # show_smiled_tilted_frame()

    # apply_frame_correction(0)
    # apply_frame_correction(1)
    # show_desmiled_intr()
    # show_desmiled_lut()
