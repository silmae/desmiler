"""

Some control over sessions to create metadata and
correct folder structure.

"""

import os

from core import properties as P
from utilities import file_handling as F
from core.camera_interface import CameraInterface
import logging
import toml
from toml import TomlDecodeError

class ScanningSession:

    # Session log? At least some metadata (datetime, camera settings etc.) should be saved I think.

    def __init__(self, session_name:str, cami:CameraInterface):
        print(f"Creating session '{session_name}'")
        self.session_name = session_name
        self.session_root = P.path_rel_scan + '/' + session_name
        self.camera_setting_path = self.session_root + '/' + P.camera_settings_file_name
        self.scan_settings_path = os.path.abspath(self.session_root + '/' + P.scan_settings_file_name)
        self._cami = cami
        self.scan_settings = None

        self.dark_path  = os.path.abspath(self.session_root + '/' + P.extension_dark + '.nc')
        self.white_path = os.path.abspath(self.session_root + '/' + P.extension_white + '.nc')
        self.light_path = os.path.abspath(self.session_root + '/' + P.extension_light + '.nc')
        self.dark = None
        self.white = None
        self.light = None

        if self.session_exists():
            print(f"Found existing session.")

            # load and apply camera settings before scan settings so that
            # the scanning can be controlled by scan settings toml, which
            # overwrite the camera settings
            logging.info(f"Searching for existing camera settings from '{self.camera_setting_path}'")
            if os.path.exists(self.camera_setting_path):
                print(f"Found existing camera settings")
                self._cami.load_camera_settings(self.camera_setting_path)

            logging.info(f"Searching for existing scan settings from '{self.scan_settings_path}'")
            if os.path.exists(self.scan_settings_path):
                print(f"Found existing scan settings")
                try:
                    with open(self.scan_settings_path, 'r') as file:
                        self.scan_settings = toml.load(file)
                    print(self.scan_settings)
                except TypeError as te:
                    logging.error(te)
                except TomlDecodeError as tde:
                    logging.error(tde)


            logging.info(f"Searching for existing dark frame from '{self.dark_path}'")
            if os.path.exists(self.dark_path):
                print(f"Found existing dark frame. Loading into memory")

            logging.info(f"Searching for existing white frame from '{self.white_path}'")
            if os.path.exists(self.white_path):
                print(f"Found existing white frame. Loading into memory")

            logging.info(f"Searching for existing light frame from '{self.light_path}'")
            if os.path.exists(self.light_path):
                print(f"Found existing light frame. Loading into memory")

        else:
            F.create_directory(self.session_root)

    def session_exists(self) -> bool:
        if os.path.exists(self.session_root):
            return True
        else:
            return False

    def close(self):
        """Turn camera off and save all stuff."""

        self._cami.turn_off()
        self._cami.save_camera_settings(self.camera_setting_path)

    def shoot_dark(self):
        """Shoots and saves a dark frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.extension_dark)

    def shoot_white(self):
        """Shoots and saves a white frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.extension_white)

    def shoot_light(self):
        """Shoots and saves a peaky light frame.

        The frame is acquired with full sensor size. The old frame size is
        resumed after acquisition.
        """

        self._shoot_reference(P.extension_light)

    def _shoot_reference(self, ref_type:str):

        if ref_type in (P.extension_dark, P.extension_white, P.extension_light):
            logging.debug(f"Crop before starting to shoot {ref_type}:\n {self._cami.get_crop_meta_dict()}")
            old, _ = self._cami.crop(full=True)
            logging.debug(f"New crop:\n {self._cami.get_crop_meta_dict()}")
            ref_frame = self._cami.get_frame_opt(count=P.dwl_default_count, method=P.dwl_default_method)
            meta_dict = self._cami.get_crop_meta_dict()
            self._cami.crop(*old)
            logging.debug(f"Reverted back to crop:\n {self._cami.get_crop_meta_dict()}")
            if ref_type == P.extension_dark:
                self.dark = ref_frame
            if ref_type == P.extension_white:
                self.white = ref_frame
            if ref_type == P.extension_light:
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

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        self._cami.crop(width, width_offset, height, height_offset, full)

