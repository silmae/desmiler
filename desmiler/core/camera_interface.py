"""

Provides camera interface to be used elswhere in the program.
Should be easier to swap to a different backend if desired.

This is very simplistic usage of camazing package. We encourage you
to play around and explore its capabilities.

############# CAMAZING AND MATRIX VISION ##############

If using camazing, you will need to install MatrixVision:

Go to http://static.matrix-vision.com/mvIMPACT_Acquire/
Choose latest version directory
Download mvGenTL_Acquire-x86_64-2.39.0.exe for 64 bit OS
Execute and follow installation wizard instructions
Windows start menu should now have vxPropView(x64) - run it. You
should be able to connect to the camera there and start frame acquisition.
In your own code (essentially, this file) initalize camera list by providing
path to the correct .cti file like so:
cameras = CameraList('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

########################################################

"""

import logging
import xarray as xr
from xarray import DataArray

from camazing import CameraList
from core import properties as P
from camazing.feature_types import AccessModeError
from genicam2.genapi import OutOfRangeException

class CameraInterface:

    """
    TODO should do a setting dump and load possibility for scanning sessions?

    """

    # Should not be used directly unless it can't be avoided.
    _cam = None

    def __init__(self):
        print(f"Initializing camera interface")
        self._initCam()

    def turn_on(self):
        logging.debug("Turning camera on.")
        self._cam.start_acquisition()

    def turn_off(self):
        logging.debug("Turning camera off.")
        self._cam.stop_acquisition()

    def get_frame(self) -> DataArray:
        return self._cam.get_frame()

    def exposure(self, value=None) -> int:
        """Set or print exposure"""
        if value is None:
            return self._cam['ExposureTime'].value
        else:
            self._set_camera_feature('ExposureTime', value)

    def gain(self, value=None) -> int:
        """Set or print gain"""
        if value is None:
            return self._cam['Gain'].value
        else:
            self._set_camera_feature('Gain', value)

    def width(self, value=None) -> int:
        """Set or print width"""
        if value is None:
            return self._cam['Width'].value
        else:
            self._set_camera_feature('Width', value)

    def width_offset(self, value=None) -> int:
        """Set or print width_offset"""
        if value is None:
            return self._cam['OffsetX'].value
        else:
            self._set_camera_feature('OffsetX', value)

    def height(self, value=None) -> int:
        """Set or print height"""
        if value is None:
            return self._cam['Height'].value
        else:
            self._set_camera_feature('Height', value)

    def height_offset(self, value=None) -> int:
        """Set or print height_offset"""
        if value is None:
            return self._cam['OffsetY'].value
        else:
            self._set_camera_feature('OffsetY', value)

    def _initCam(self):
        """Initializes the camera.

        Assumes only one camera is connected to the machine so the first one
        in camazing's CameraList will be used.

        Camera settings are loaded from 'camera_settings.toml'

        """

        cameras = CameraList('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

        if len(cameras) < 1:
            raise RuntimeError("Could not find the camera. Camera could not be initialized.")

        self._cam = cameras[0]

        try:
            logging.info("Initializing camera...", end=' ')
            self._cam.initialize()
            logging.info("done")
        except RuntimeError as re:
            raise RuntimeError("Could not initialize the camera. Runtime error.") from re
        except FileNotFoundError as fnf:
            raise RuntimeError("Could not initialize the camera. FileNotFoundError.") from fnf
        except Exception:
            raise RuntimeError("Could not initialize the camera. You may have another instance using the camera.")

        # Read camera settings from a file.
        _, errors = self._cam.load_config_from_file(P.camera_settings_path)

        if len(errors) != 0:
            logging.warning(f"Errors provided by camazing when loading settings from path {P.camera_settings_path}:")
        for e in enumerate(errors):
            logging.warning(e)

    def _set_camera_feature(self, name, val):
        """Change camera settings.

        Can be called even if LiveFeed is running.

        First, we try to set the given feature without pausing the feed,
        and pause only if that fails. Possible exceptions are printed to the
        console.

        TODO Changing of width, height, OffsetX, and OffsetY is not allowed. Use crop() instead.

        Note: OutOfRangeException related to features with certain increment
        (e.g. height, width) throws an exception which I cannot handle here.

        Parameters
        ----------
        name : string
            Name of the feature e.g. 'ExposureTime'
        val :
            Value to be set. Depending on the feature this might
            be int, string, bool etc.
        """


        # if name in ["Height", "Width", "OffsetX", "OffsetY"]:
        #     print("Use method crop() instead.")

        if name in self._cam:
            cam_was_acquiring = self._cam.is_acquiring()
            try:
                # Try to set the value even if live feed is running.
                self._cam[name].value = val
                print(f"Feature \'{name}\' was succesfully set to {val}.")
            except AccessModeError:
                logging.warning(f"Could not change the feature {name} on the fly. Pausing and trying again.")
                self._cam.stop_acquisition()
                try:
                    self._cam[name].value = val
                    print(f"Feature \'{name}\' was succesfully set to {val}.")
                except AccessModeError as ame:
                    # Could not set the value even with feed paused.
                    logging.errror(ame)
            except ValueError as ve:
                # Catch value error from both of the previous cases.
                logging.error(ve)
            except OutOfRangeException as ore:
                # FIXME Catching this exception doesn't work properly.
                # It might be that camazing is not throwing it as it should.
                logging.error(f"Increment exception probably: {ore}")
                # print(ore)
            except:
                logging.error(f"Unexpected exception while trying to set the feature {name}")
            finally:
                # Try to unpause the animation even if exceptions occurred.
                if cam_was_acquiring:
                    self._cam.start_acquisition()
        else:
            logging.warning(f"Feature '{name}' is not valid. Try again with valid name.")
