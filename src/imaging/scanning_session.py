"""

Some control over sessions to create metadata and
correct folder structure.

Order:

    if new session:
        - generate default control file and close it
        - copy default camera settings from main directory and close it
        - both files can now be edited and saved before imaging
        - load both control files when initializing the camera
        - shoot (dark, white, light) images and save to session directory
        - start scanning
    if existing session:
        - load control files
        - load existing dark, white, light frames
        - start scanning (TODO how to deal with multiple scans in the same session?)

"""

import logging
import toml
from toml import TomlDecodeError
import numpy as np
import os
import xarray as xr
from xarray import Dataset

from core import properties as P
from utilities import file_handling as F
from core.camera_interface import CameraInterface
from core import smile_correction as sc
import core.frame_manipulation as fm
import core.cube_manipulation as cm
from analysis.cube_inspector import CubeInspector
import analysis.frame_inspector as fi
import time
import math
import matplotlib.pyplot as plt

import analysis.cube_inspector as ci

class ScanningSession:

    def __init__(self, session_name:str):
        """Initializes new scanning session with given name.

        Creates a default directory (/scans/session_name) if it does not yet exist. If it
        exists, the existing files are loaded.

        Parameters
        ----------
        session_name
            Name of the session.
        """

        self.session_name = session_name
        self.session_root = P.path_rel_scan + '/' + session_name + '/'
        self.camera_setting_path = self.session_root + P.fn_camera_settings
        self.scan_settings_path = os.path.abspath(self.session_root + P.fn_control)
        self.dark_path  = os.path.abspath(self.session_root + P.ref_dark_name + '.nc')
        self.white_path = os.path.abspath(self.session_root + P.ref_white_name + '.nc')
        self.light_path = os.path.abspath(self.session_root + P.ref_light_name + '.nc')
        self.shift_path = os.path.abspath(self.session_root + P.shift_name + '.nc')
        self.cube_raw_path = os.path.abspath(self.session_root + P.cube_raw_name + '.nc')
        self.cube_rfl_path = os.path.abspath(self.session_root + P.cube_reflectance_name + '.nc')
        self.cube_desmiled_lut_path = os.path.abspath(self.session_root + P.cube_desmiled_lut + '.nc')
        self.cube_desmiled_intr_path = os.path.abspath(self.session_root + P.cube_desmiled_intr + '.nc')

        self._cami = None
        self.control = None
        self.dark = None
        self.white = None
        self.light = None

        if self.session_exists():
            print(f"Found existing session '{session_name}'.")

            logging.info(f"Searching for existing dark frame from '{self.dark_path}'")
            if os.path.exists(self.dark_path):
                print(f"Found existing dark frame. Loading into memory", end='...')
                self.dark = F.load_frame(self.dark_path)
                print("done")

            logging.info(f"Searching for existing white frame from '{self.white_path}'")
            if os.path.exists(self.white_path):
                print(f"Found existing white frame. Loading into memory", end='...')
                self.white = F.load_frame(self.white_path)
                print("done")

            logging.info(f"Searching for existing light frame from '{self.light_path}'")
            if os.path.exists(self.light_path):
                print(f"Found existing light frame. Loading into memory", end='...')
                self.light = F.load_frame(self.light_path)
                print("done")

        else:
            print(f"Creating new session '{session_name}'")
            F.create_directory(self.session_root)

        if not os.path.exists(self.scan_settings_path):
            self.generate_default_scan_control()
        else:
            self.load_control_file()
        print(f"Session initialized and ready to work.")

    def __del__(self):
        """Delete object after calling close() for cleanup."""

        self.close()

    def close(self):
        """Turn camera off and save camera settings."""

        if self._cami is not None:
            self._cami.turn_off()
            self._cami.save_camera_settings(self.camera_setting_path)
            del self._cami

    def _init_cami(self):
        """Initialize CameraInterface if not initialized yet."""

        if self._cami is None:
            self._cami = CameraInterface()
            self._cami.save_camera_settings(self.camera_setting_path)

    def reload_settings(self):
        """Reload all settings.

        Use this after making changes to camera settings file or the control file.
        """

        self.load_camera_settings()

    def load_camera_settings(self):
        """Load camera settings from a file.

        Also loads control file (possibly again) as it must override some camera settings,
        such as cropping.
        """

        if self._cami is None:
            self._cami = CameraInterface()

        # load and apply camera settings before scan settings so that
        # the scanning can be controlled by scan settings toml, which
        # overwrite the camera settings
        logging.info(f"Searching for existing camera settings from '{self.camera_setting_path}'")
        if os.path.exists(self.camera_setting_path):
            print(f"Loading camera settings")
            self._cami.load_camera_settings(self.camera_setting_path)
            print(f"Camera settings loaded and applied.")

        # Load control after camera initialization to override
        self.load_control_file()

    def load_control_file(self):
        self.control = F.load_control_file(self.scan_settings_path)

    def session_exists(self) -> bool:
        if os.path.exists(self.session_root):
            return True
        else:
            return False

    def shoot_dark(self):
        """Shoots and saves a dark frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.ref_dark_name)

    def shoot_white(self):
        """Shoots and saves a white frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.ref_white_name)

    def shoot_light(self):
        """Shoots and saves a peaky light frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.ref_light_name)

    def _show_reference(self, ref_type:str):

        if ref_type in (P.ref_dark_name, P.ref_white_name, P.ref_light_name):

            if ref_type == P.ref_dark_name:
                ref_frame = self.dark
            if ref_type == P.ref_white_name:
                ref_frame = self.white
            if ref_type == P.ref_light_name:
                ref_frame = self.light
            fi.plot_frame(ref_frame)
        else:
            logging.error(f"Wrong reference type '{ref_type}'")

    def _shoot_reference(self, ref_type:str):
        """Shoot a reference frame (dark, white or light) and save to disk."""

        self._init_cami()
        self.reload_settings()
        if ref_type in (P.ref_dark_name, P.ref_white_name, P.ref_light_name):
            logging.debug(f"Crop before starting to shoot {ref_type}:\n {self._cami.get_crop_meta_dict()}")
            old, _ = self._cami.crop(full=True)
            logging.debug(f"New crop:\n {self._cami.get_crop_meta_dict()}")
            print(f"Shooting frame (avg of {P.dwl_default_count} with {self._cami.exposure():.1f} micro seconds.)")
            ref_frame = self._cami.get_frame_opt(count=P.dwl_default_count, method=P.dwl_default_method)
            meta_dict = self._cami.get_crop_meta_dict()
            ref_frame = F.save_frame(ref_frame, self.session_root + '/' + ref_type, meta_dict=meta_dict)
            self._cami.crop(*old)
            logging.debug(f"Reverted back to crop:\n {self._cami.get_crop_meta_dict()}")
            if ref_type == P.ref_dark_name:
                self.dark = ref_frame
            if ref_type == P.ref_white_name:
                self.white = ref_frame
            if ref_type == P.ref_light_name:
                self.light = ref_frame
            self._show_reference(ref_type)
        else:
            logging.error(f"Wrong reference type '{ref_type}'")

    def run_scan(self):
        """Run a scan as defined in the control file of current session.

        """

        self.reload_settings()
        width = self.control[P.ctrl_scan_settings][P.ctrl_width]
        width_offset = self.control[P.ctrl_scan_settings][P.ctrl_width_offset]
        height = self.control[P.ctrl_scan_settings][P.ctrl_height]
        height_offset = self.control[P.ctrl_scan_settings][P.ctrl_height_offset]
        mock = self.control[P.ctrl_scan_settings][P.ctrl_is_mock_scan]
        speed = self.control[P.ctrl_scan_settings][P.ctrl_scanning_speed_value]
        length = self.control[P.ctrl_scan_settings][P.ctrl_scanning_length_value]
        # exposure_time_ms = self._cami.exposure()
        # exposure_time_mu_s = self.control[P.ctrl_scan_settings][P.ctrl_exporure_time]
        exposure_time_s = self.control[P.ctrl_scan_settings][P.ctrl_exporure_time_s]
        overhead = self.control[P.ctrl_scan_settings][P.ctrl_acquisition_overhead]

        # fps = 1 / (exposure_time_s * (1+exposure_overhead))
        # scan_time_raw = length / speed
        # scan_time_w_overhead = scan_time_raw * (1+exposure_overhead)
        # frame_count = int(scan_time_raw * fps)
        # frame_time = 1 / fps

        time_total = length / speed
        time_frame = exposure_time_s * (overhead + 1)
        frame_count = math.ceil(time_total / time_frame)
        fps = 1 / time_frame
        aspect_ratio = frame_count / height
        aspect_ratio_s = f"{frame_count}/{height}"

        rjust = 30

        #FIXME redo time calculation! we cannot change the time it takes to travel as it is restricted
        # by the scanning platform.

        if mock:
            print(f"Running a mock scan. Just printing you the parameters etc., but not recording data.")
        else:
            print(f"Starting a scan")

        print("Scanning parameters from control file:")
        for key, val in self.control[P.ctrl_scan_settings].items():
            print(f"'{key}':".rjust(rjust) + f"\t{val}")

        print(f"Derived values:")
        print(f"Scanning duration:".rjust(rjust) + f"\t {time_total:.3f} s")
        print(f"Frame count:".rjust(rjust) + f"\t {frame_count}")
        print(f"FPS:".rjust(rjust) + f"\t {fps}")
        print(f"Aspect ratio (w/h):".rjust(rjust) + f"\t {aspect_ratio_s} ({aspect_ratio})")

        if not mock:
            if self._cami is None:
                self._init_cami()
            self._cami.turn_on()
            self._cami.exposure(exposure_time_s * 1e6)
            self._cami.crop(width, width_offset, height, height_offset, full=False)
            frame_list = [None]*frame_count
            frame_time_list = np.zeros((frame_count,), dtype=np.float64)
            wait_time_list = np.zeros((frame_count,), dtype=np.float64)

            # Get one frame before starting the loop as it will take more time
            # than the rest. Probably because some initializations of the camera.
            # This temporary frame will be overwritten in the loop.
            frame_list[0] = self._cami.get_frame()

            print(f"ExposureMode: {self._cami._cam['ExposureMode'].value}")
            print(f"ExposureAuto: {self._cami._cam['ExposureAuto'].value}")
            print(f"pgrExposureCompensationAuto: {self._cami._cam['pgrExposureCompensationAuto'].value}")

            ### Estimate frame time ###

            # test_count = 20
            # print(self._cami.exposure())
            # for i in range(test_count):
            #     time_start = time.perf_counter()
            #     f = self._cami.get_frame()
            #     stop = time.perf_counter()
            #     time_elapsed = stop - time_start
            #     f.coords[P.dim_scan] = i
            #     frame_list[i] = f.copy(deep=True)
            #
            #     frame_time_list[i] = time_elapsed
            #
            # print(self._cami.exposure())
            # avg_frame_time = np.mean(frame_time_list[0:test_count])
            # std_frame_time = np.std(frame_time_list[0:test_count])
            #
            # plt.plot(frame_time_list[0:test_count])
            # plt.show()
            #
            # print(f"Average frame time was {avg_frame_time:.4f} (+- {std_frame_time:.6f}) s "
            #       f"while prior estimation was {time_frame:.4f} s.")
            #
            # time_frame = avg_frame_time * (overhead+1)
            # frame_count = math.ceil(time_total / time_frame)
            # fps = 1 / time_frame
            #
            # print(f"Adjusted frame count:".rjust(rjust) + f"\t {frame_count}")
            # print(f"Adjusted FPS:".rjust(rjust) + f"\t {fps}")

            ### Estimate frame time end ###

            frame_list = [None]*frame_count
            frame_time_list = np.zeros((frame_count,), dtype=np.float64)
            wait_time_list = np.zeros((frame_count,), dtype=np.float64)

            scan_start_time = time.perf_counter()
            print(f"Scan start with exposure {self._cami.exposure()}")

            for i in range(frame_count):
                time_start = time.perf_counter()
                f = self._cami.get_frame()
                time_elapsed = (time.perf_counter() - time_start)
                f.coords[P.dim_scan] = i
                # frame_list.append(f)
                frame_list[i] = f.copy(deep=True)

                wait_time = max(time_frame - time_elapsed, 0.0)
                frame_time_list[i] = time_elapsed
                wait_time_list[i] = wait_time

                if wait_time > 0.:
                    time.sleep(wait_time)

            print("Scan done")

            print(f"Total scan duration {((time.perf_counter() - scan_start_time)):.3f} s.")

            avg_frame_time = np.mean(frame_time_list)
            std_frame_time = np.std(frame_time_list)
            avg_wait_time = np.mean(wait_time_list)
            std_wait_time = np.std(wait_time_list)
            print(f"Average frame time was {avg_frame_time:.4f} (+- {std_frame_time:.6f}) s "
                  f"while prior estimation was {time_frame:.4f} s.")
            print(f"Average wait time was {avg_wait_time:.4f} (+- {std_wait_time:.6f}) s ")

            if avg_frame_time < time_frame * 0.9:
                print(f"More than 10 % idle time. Consider reducing the "
                      f"'{P.ctrl_acquisition_overhead}' percentage.")
            elif avg_frame_time > time_frame * 1.1:
                print(f"Estimated frame time exceeded by more than 10 %. "
                      f"Consider increasing the '{P.ctrl_acquisition_overhead}' percentage.")

            # Plot exposure times for fun
            plt.plot(frame_time_list)
            plt.show()

            # TODO save metadata too
            print("Saving the raw cube")
            frames = xr.concat(frame_list, dim=P.dim_scan)
            raw_cube = xr.Dataset(data_vars={P.naming_cube_data: frames})
            F.save_cube(raw_cube, self.cube_raw_path)
            print("Cube saved")
            self._cami.turn_off()

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        """TODO this may be removed... i think. The control file should take care of cropping."""

        if self._cami is not None:
            self._cami.crop(width, width_offset, height, height_offset, full)
        else:
            logging.warning(f"Cannot crop session without a camera.")

    def generate_default_scan_control(self):
        """Generates default control file for the user to modify externally.

        Does nothing if there already exists a control file for current session.

        Default control content is defined in core.properties file.
        """

        abs_path = os.path.abspath(self.session_root + P.fn_control)

        if not os.path.exists(abs_path):
            print(f"Creating default control file to '{abs_path}'", end='')
            self.control = toml.loads(P.example_scan_control_content)

            with open(abs_path, "w") as file:
                file.write(P.example_scan_control_content)

            print(f"Default control file created.")

    def make_reflectance_cube(self) -> Dataset:
        """ Makes a reflectance cube out of a raw cube and saves onto disk.

        Returns
        -------
            Dataset
                Resulting reflectance cube.
        """

        org = F.load_cube(self.cube_raw_path)

        rfl = cm.make_reflectance_cube(org, self.dark, self.white, self.control)

        print(f"Saving reflectance cube to {self.cube_rfl_path}...", end=' ')
        F.save_cube(rfl, self.cube_rfl_path)
        print(f"done")
        return rfl

    def make_shift_matrix(self):
        """Make shift matrix and save it to disk.

        TODO move elsewhere
        """

        width = self.control[P.ctrl_scan_settings][P.ctrl_width]
        width_offset = self.control[P.ctrl_scan_settings][P.ctrl_width_offset]
        height = self.control[P.ctrl_scan_settings][P.ctrl_height]
        height_offset = self.control[P.ctrl_scan_settings][P.ctrl_height_offset]

        positions = np.array(self.control[P.ctrl_spectral_lines][P.ctrl_positions]) - width_offset
        peak_width = self.control[P.ctrl_spectral_lines][P.ctrl_peak_width]
        bandpass_width = self.control[P.ctrl_spectral_lines][P.ctrl_window_width]

        if self.light is None:
            raise RuntimeError(f"Light data does not exist. Shoot one using ui.shoot_light(). "
                          f"Aborting shift matrix generation. ")

        light_ds = self.light.isel({P.dim_x: slice(width_offset, width_offset + width),
                                  P.dim_y: slice(height_offset, height_offset + height)})
        light_frame = light_ds[P.naming_frame_data]
        bp = sc.construct_bandpass_filter(light_frame, positions, bandpass_width)
        sl_list = sc.construct_spectral_lines(light_frame, positions, bp)
        shift_matrix = sc.construct_shift_matrix(sl_list, light_frame[P.dim_x].size, light_frame[P.dim_y].size)
        # frame_inspector.plot_frame(light_frame, sl_list, True, True, False, 'testing')

        abs_path = os.path.abspath(self.shift_path)
        print(f"Saving shift matrix to {abs_path}...", end=' ')
        shift_matrix.to_netcdf(abs_path)
        print("done")
        # Uncomment for debugging
        # shift_matrix.plot.imshow()
        # plt.show()
        return shift_matrix, sl_list

    def desmile_cube(self, source_cube=None, shift_method=0) -> Dataset:
        """ Desmile a reflectance cube with LUT or INTR shifts and save and return the result.

        Load the reflectance cube from default path. If you want to desmile raw cube, just
        load it separately and pass as parameter.

        Parameters
        ----------
            source_cube : Dataset
                Optional. If not given, the default reflectance cube for current session is loaded.
                This is the recommended usage and passing a cube implicitly is for special cases.
            shift_method : int
                Shift method 0 is for lookup table shift and 1 for interpolative shift. Default is 0.
        Returns
        -------
            Dataset
                Desmiled cube for chaining.
        """


        if os.path.exists(os.path.abspath(self.shift_path)):
            logging.info(f"Desmiling with existing shift matrix from '{self.shift_path}'.")
            shift = F.load_shit_matrix(self.shift_path)
        else:
            logging.info(f"Generating new shift matrix for desmiling.")
            shift, _ = self.make_shift_matrix()

        print("This how your shift matrix looks like. Close the window to continue.")
        shift.plot()
        plt.show()

        if shift_method == 0:
            cube_type = 'lut'
            save_path = self.cube_desmiled_lut_path
        elif shift_method == 1:
            cube_type = 'intr'
            save_path = self.cube_desmiled_intr_path

        if source_cube is None:
            rfl = F.load_cube(self.cube_rfl_path)
        else:
            rfl = source_cube

        print(f"Desmiling with {cube_type} shifts...", end=' ')
        desmiled = rfl.copy(deep=True)
        del rfl
        desmiled = sc.apply_shift_matrix(desmiled, shift, method=shift_method, target_is_cube=True)
        print(f"done")

        print(f"Saving desmiled cube to {save_path}...", end=' ')
        F.save_cube(desmiled, save_path)
        print(f"done")
        return desmiled

    def show_cube(self, force_raw_cube=False):
        try:
            if os.path.exists(self.cube_rfl_path) and not force_raw_cube:
                target_cube = F.load_cube(self.cube_rfl_path)
                viewable = P.naming_reflectance
            elif os.path.exists(self.cube_raw_path):
                target_cube = F.load_cube(self.cube_raw_path)
                viewable = P.naming_cube_data
            else:
                raise RuntimeError(f"No source cube to show.")
            if os.path.exists(self.cube_desmiled_lut_path):
                target_cube_2 = F.load_cube(self.cube_desmiled_lut_path)
            else:
                logging.warning(f"LUT desmiled cube does not exist. Using original cube instead.")
                target_cube_2 = target_cube
            if os.path.exists(self.cube_desmiled_lut_path):
                target_cube_3 = F.load_cube(self.cube_desmiled_intr_path)
            else:
                logging.warning(f"INTR desmiled cube does not exist. Using original cube instead.")
                target_cube_3 = target_cube

            ci = CubeInspector(target_cube, target_cube_2, target_cube_3, viewable=viewable, control=self.control)
            ci.show()
        except FileNotFoundError as fnf:
            logging.error(fnf)
            print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")
        except RuntimeError as r:
            logging.error(r)
            print(f"Could not load one of the cubes. Run synthetic_data.generate_cube_examples() and try again.")

    def show_shift(self):
        s = F.load_shit_matrix(self.shift_path)
        s.plot()
        plt.show()

    def show_light(self):
        # fi.plot_frame(F.load_frame(self.light_path))
        self.reload_settings()
        shift, sl = self.make_shift_matrix()
        fi.plot_frame(self.light, spectral_lines=sl, plot_circ_fit=True, plot_fit_points=True, control=self.control)

    def exposure(self, value=None) -> int:
        if self._cami is not None:
            return self._cami.exposure(value)
        else:
            logging.warning(f"Cannot set exposure as CameraInterface does not exist.")

def create_example_scan():
    """Creates an example if does not exist."""
    # TODO if does not exist
    example_sc = ScanningSession(P.example_scan_name)
    example_sc.generate_default_scan_control()


