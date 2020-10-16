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
import numpy as np
import os
from xarray import Dataset

from core import properties as P
from utilities import file_handling as F
from core.camera_interface import CameraInterface
from core import smile_correction as sc
import core.frame_manipulation as fm
import core.cube_manipulation as cm

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

    def _shoot_reference(self, ref_type:str):
        """Shoot a reference frame (dark, white or light) and save to disk."""

        self._init_cami()
        if ref_type in (P.ref_dark_name, P.ref_white_name, P.ref_light_name):
            logging.debug(f"Crop before starting to shoot {ref_type}:\n {self._cami.get_crop_meta_dict()}")
            old, _ = self._cami.crop(full=True)
            logging.debug(f"New crop:\n {self._cami.get_crop_meta_dict()}")
            ref_frame = self._cami.get_frame_opt(count=P.dwl_default_count, method=P.dwl_default_method)
            meta_dict = self._cami.get_crop_meta_dict()
            self._cami.crop(*old)
            logging.debug(f"Reverted back to crop:\n {self._cami.get_crop_meta_dict()}")
            if ref_type == P.ref_dark_name:
                self.dark = ref_frame
            if ref_type == P.ref_white_name:
                self.white = ref_frame
            if ref_type == P.ref_light_name:
                self.white = ref_frame

            F.save_frame(ref_frame, self.session_root + '/' + ref_type, meta_dict=meta_dict)
        else:
            logging.error(f"Wrong reference type '{ref_type}'")

    def run_scan(self, mock=True):
        """Run a scan as defined in the control file of current session.

        Parameters
        ----------
        mock : bool
            If True, no actual work is done. This is a dry run for checking parameters etc..

        """

        width = self.control['scan_settings']['width']
        width_offset = self.control['scan_settings']['width_offset']
        height = self.control['scan_settings']['height']
        height_offset = self.control['scan_settings']['height_offset']
        if mock:
            print(f"Running a mock scan. Just printing you the parameters etc. but not recording.")
            print("Scanning parameters from control file:")
            for key, val in self.control[P.ctrl_scan_settings].items():
                print(f"\t'{key}': \t{val}")
            self.crop(width, width_offset, height, height_offset)
        else:
            self._init_cami()
            # TODO implement

    def sweep(self, length=0.5, vel=0.01, count=None, dryRun=False, fileName=None):
        """Capture eithr a linear or angular sweep.

        Returns the resulting spectral cube, and saves it if a fileName is given.
        Prints a countdown before recording is started.
        Dark frame is not substracted and desmiling is not performed.

        Parameters
        ----------
            length : float
                Total travel distance of the camera in meters or angle in degrees.
            vel : float
                Velocity or angular velocity in meters per second or in degrees per second.
            count : int, optional, default None
                How many frames are captured. If none given, height of a frame is used
                resulting in aspect ratio 1:1.
            dryRun : bool, default False
                If True, actual frame capturing is not done, but parameters for recording
                are printed.
            fileName : str, default None
                Save resulting spectral cube with this name. Handle to file is closed.
                If None, the result is not saved.
        Returns
        -------
            DataSet
                Resulting dataset is returned.
        """

        print("Trying to do a sweep. Implement me please!")

        # if not self._cam.is_acquiring():
        #     self._cam.start_acquisition()
        #
        # # Make x-y-ratio 1:1 as default for now
        # if count is None:
        #     count = self._cam['Height'].value
        #
        # sweepTime = length / vel
        # print(f"Total sweep time: {sweepTime} s")
        # # Frame time in micro seconds
        # frameTime = 1000000 * (sweepTime / count)
        # print(f"Frame time {frameTime} us and exposure time {self._cam['ExposureTime'].value} us")
        #
        # if not dryRun:
        #     # Save consecutive frames into a list.
        #     frameList = []
        #
        #     print(f"Shooting {count} frames...", end=' ')
        #     # Countdown
        #     for i in range(3):
        #         print(3 - i, end=', ')
        #         time.sleep(0.5)
        #     print("Go!")
        #
        #     sweepStartTime = time.perf_counter_ns()
        #     for i in range(count):
        #         tStart = time.perf_counter_ns()
        #         f = self._cam.get_frame()
        #         # f = ft.subtractDark(frame=f, dark=self._darkFrame)
        #         f.coords['index'] = i
        #         frameList.append(f)
        #
        #         # Whole processing time of a single frame in micro seconds
        #         duration_us = (time.perf_counter_ns() - tStart) / 1000
        #         tw = frameTime - duration_us
        #         # print(f"Should wait for {tw} microsecs.")
        #         if tw > 0:
        #             time.sleep(tw / 1000000)  # in seconds
        #
        #     print(f"..done. Total sweep duration {((time.perf_counter_ns() - sweepStartTime) / 1000000000):.3f} s.")
        #     print("Starting to process...", end=' ', flush=True)
        #     tProsStart = time.perf_counter_ns()
        #     frames = xr.concat(frameList, dim='index')
        #     ds = xr.Dataset(
        #         data_vars={
        #             'dn': frames,
        #             'x_shift': self._desmileArray,
        #         },
        #     )

            # NOTE desmiling should be done separately so that the original raw cube stays intact
            # ds['desmiled_x'] = ds.x_shift + ds.x
            # ds.coords['new_x'] = np.linspace(ds.desmiled_x.min(), ds.desmiled_x.max(), 1000)
            # ds = ds.groupby('y').apply(self.desmile_row)
            # duration_s = (time.perf_counter_ns() - tProsStart) / 1000000000
            # print(f"..done. Elapsed time {duration_s} s")

            # if fileName is not None:
            #     path = f"cubes/{fileName}.nc"
            #     print(f"Saving cube to path '{path}'...", end=' ')
            #     ds.to_netcdf(os.path.normpath(path))
            #     ds.close()
            #     print("done.")
            #
            # return ds

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
                toml.dump(self.control, file)

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

        s = F.load_shit_matrix(self.session_root + P.shift_name)

        print(f"Desmiling with {cube_type} shifts...", end=' ')
        desmiled = rfl.copy(deep=True)
        del rfl
        desmiled = sc.apply_shift_matrix(desmiled, s, method=shift_method, target_is_cube=True)
        print(f"done")

        print(f"Saving desmiled cube to {save_path}...", end=' ')
        F.save_cube(desmiled, save_path)
        print(f"done")
        return desmiled


def create_example_scan():
    """Creates an example if does not exist."""
    # TODO if does not exist
    example_sc = ScanningSession(P.example_scan_name)
    example_sc.generate_default_scan_control()


