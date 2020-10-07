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

import os

from core import properties as P
from utilities import file_handling as F
from core.camera_interface import CameraInterface
import logging
import toml

class ScanningSession:

    def __init__(self, session_name:str):
        print(f"Creating session '{session_name}'")
        self.session_name = session_name
        self.session_root = P.path_rel_scan + '/' + session_name
        self.camera_setting_path = self.session_root + '/' + P.fn_camera_settings
        self.scan_settings_path = os.path.abspath(self.session_root + '/' + P.fn_control)
        self._cami = None
        self.scan_settings = None

        self.dark_path  = os.path.abspath(self.session_root + '/' + P.ref_dark_name + '.nc')
        self.white_path = os.path.abspath(self.session_root + '/' + P.ref_white_name + '.nc')
        self.light_path = os.path.abspath(self.session_root + '/' + P.ref_light_name + '.nc')
        self.dark = None
        self.white = None
        self.light = None

        if self.session_exists():
            print(f"Found existing session.")

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
            F.create_directory(self.session_root)

        if not os.path.exists(self.scan_settings_path):
            self.generate_default_scan_control()
        else:
            self.load_control_file()

    def __del__(self):
        self.close()

    def _init_cami(self):
        if self._cami is None:
            self._cami = CameraInterface()

    def reload_settings(self):
        self.load_camera_settings()

    def load_camera_settings(self):

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
        self.scan_settings = F.load_control_file(self.scan_settings_path)

    def session_exists(self) -> bool:
        if os.path.exists(self.session_root):
            return True
        else:
            return False

    def close(self):
        """Turn camera off and save all stuff."""

        if self._cami is not None:
            self._cami.turn_off()
            self._cami.save_camera_settings(self.camera_setting_path)
            del self._cami

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
        width = self.scan_settings['scan_settings']['width']
        width_offset = self.scan_settings['scan_settings']['width_offset']
        height = self.scan_settings['scan_settings']['height']
        height_offset = self.scan_settings['scan_settings']['height_offset']
        if mock:
            print(f"Running a mock scan. Just printing you the parameters etc. but not recording.")
            print("Scanning parameters from control file:")
            for key, val in self.scan_settings['scan_settings'].items():
                print(f"\t'{key}': {val}")
            self.crop(width, width_offset, height, height_offset)
        else:
            self._init_cami()
            # TODO implement

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        """FIXME this may be removed... i think. The control file should take care of cropping."""

        if self._cami is not None:
            self._cami.crop(width, width_offset, height, height_offset, full)
        else:
            logging.warning(f"Cannot crop session without a camera.")

    def generate_default_scan_control(self, path=None):
        if path is None:
            path = self.session_root + '/' + P.fn_control
            abs_path = os.path.abspath(path)
        else:
            abs_path = os.path.abspath(path + '/' + P.fn_control)

        if not os.path.exists(abs_path):
            print(f"Creating default control file to '{abs_path}'", end='')
            control_toml_s = toml.loads(P.example_scan_control_content)

            with open(abs_path, "w") as file:
                toml.dump(control_toml_s, file)

            print(f"Default control file created.")

def create_example_scan():
    """Creates an example if does not exist."""

    example_sc = ScanningSession(P.example_scan_name)
    example_sc.generate_default_scan_control()


