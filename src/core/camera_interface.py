"""

Provides camera interface to be used elsewhere in the program.
Should be easier to swap to a different backend if desired.

This is very simplistic usage of the camazing library. We encourage you
to play around and explore its capabilities.

############# CAMAZING AND MATRIX VISION ##############

If using camazing, you will need to install MatrixVision:

Go to http://static.matrix-vision.com/mvIMPACT_Acquire/
Choose latest version directory
Download mvGenTL_Acquire-x86_64-X.XX.X.exe for 64 bit OS
Execute and follow installation wizard instructions
Windows start menu should now have vxPropView(x64) - run it. You
should be able to connect to the camera there and start frame acquisition
(press 'Use' and then 'Acquire').
In your own code (essentially, this file) initialize camera list by providing
path to the correct .cti file like so:
cameras = CameraList('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

########################################################

"""

import logging
import xarray as xr
from xarray import DataArray
import os

from camazing import CameraList
from core import properties as P
from utilities import numeric as N
from camazing.feature_types import AccessModeError
from genicam2.genapi import OutOfRangeException


cti_path = 'C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti'

class CameraInterface:
    """Camera interface class for frame grabbing.com

    Provides access to some of the most important camera settings, such as exposure time.

    """

    # Should not be used directly unless it can't be avoided.
    _cam = None

    def __init__(self):
        """Initializes the camera.

        Assumes only one camera is connected to the machine so the first one
        in camazing's CameraList will be used.

        Camera settings are loaded from 'camera_settings.toml' (top level directory).

        Logs and reraises any errors raised by camazing.

        Raises
        ------
        RuntimeError
            If no camera is connected to the machine.
        """

        logging.debug(f"Initializing camera interface")

        try:
            cameras = CameraList(cti_path)
        except:
            logging.fatal(f"Camera list could not be initialized. Make sure that cti_path in "
                          f"camera_interfase.py points to a correct file. If you don't know what this "
                          f"means, read documentation in core/camera_interfase.py.")
            raise

        if len(cameras) < 1:
            raise RuntimeError("Could not find the camera (cameralist length 0). "
                               "Make sure the camera is connected to the machine and the USB cable is "
                               "properly set. Camera could not be initialized.")

        self._cam = cameras[0]

        try:
            logging.debug("Initializing camera...")
            self._cam.initialize()
            logging.debug("done")
        except RuntimeError as re:
            logging.error(re)
            raise
        except FileNotFoundError as fnf:
            logging.error(fnf)
            raise
        except Exception:
            logging.error("Could not initialize the camera. You may have another instance using the camera.")
            raise

        # Read camera settings from a file.
        abs_path = P.path_rel_default_cam_settings
        _, errors = self._cam.load_config_from_file(abs_path)

        if len(errors) != 0:
            logging.warning(f"Errors provided by camazing when loading settings from path {abs_path}:")
            for e in enumerate(errors):
                logging.warning(f"\t{e}")

    def __del__(self):
        """Calls close() for cleanup."""

        self.close()

    def close(self):
        """Stop acquisition and delete camera."""

        if self._cam is not None:
            if self._cam.is_initialized():
                self._cam.stop_acquisition()
            del self._cam

    def turn_on(self):
        """Turn the camera on to get frames out of it. """

        logging.debug("Turning camera on.")
        self._cam.start_acquisition()

    def turn_off(self):
        """Turn the camera off. """

        if self._cam is not None and self._cam.is_initialized():
            logging.debug("Turning camera off.")
            self._cam.stop_acquisition()

    def is_on(self):
        """Returns true if camera state is 'acquiring'."""

        return self._cam.is_acquiring()

    def get_frame(self) -> DataArray:
        """Rapidly acquire a frame without any checks or options.

        Use this with scanning and live feed. Use get_frame_opt() for mean or median frames.
        Make sure camera is on with is_on() method.
        """

        return self._cam.get_frame()

    def get_frame_opt(self, count=1, method='mean') -> DataArray:
        """Acquire a frame which is a mean or median of several frames.

        Camera state (acquiring or not) will be preserved.

        Parameters
        ----------
        count : int, default=1
            If given, the mean of 'mean' consecutive frames is returned. If count == 1
            this is the same as get_frame().
        method: str, default = 'mean'
            Either 'mean' or 'median' of count consecutive frames.

        Returns
        -------
        frame : DataArray
            The shot frame.
        """

        cam_was_acquiring = self._cam.is_acquiring()
        if not self._cam.is_acquiring():
            self._cam.start_acquisition()

        if count == 1:
            return self._cam.get_frame()
            if not cam_was_acquiring:
                self._cam.stop_acquisition()
        else:
            frames = []
            for _ in range(count):
                frames.append(self._cam.get_frame())

            frame = xr.concat(frames, dim='timestamp')
            if method == 'mean':
                frame = frame.mean(dim='timestamp')
            elif method == 'median':
                frame = frame.median(dim='timestamp')
            else:
                logging.error(f"Shooting method '{method}' not recognized. Use either 'mean' or 'median'.")

            if not cam_was_acquiring:
                self._cam.stop_acquisition()

            return frame

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

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        """Change the size and position of the frame acquired from the camera.

        Retains camera's acquiring state. Stops acquisition during changes.

        Parameters
        ----------
        full : bool, optional, default=False
            If true, all other passed values are ignored and cropping is
            set to camera's full frame size with zero offset.

        Returns
        -------
        (old_vals, new_vals)
            (old_vals, new_vals) Both are 4-tuples containing (width, width_offset, height, height_offset)
        """

        was_acquiring = self._cam.is_acquiring()
        self._cam.stop_acquisition()

        w_max = self._cam['Width'].max
        h_max = self._cam['Height'].max

        # Old values
        w = self._cam['Width'].value
        w_o = self._cam['OffsetX'].value
        h = self._cam['Height'].value
        h_o = self._cam['OffsetY'].value
        old_vals = (w,w_o,h,h_o)

        if full:
            self._cam['Width'].value = w_max
            self._cam['OffsetX'].value = 0
            self._cam['Height'].value = h_max
            self._cam['OffsetY'].value = 0
        else:
            if width is None:
                width = w
            else:
                width = N.clamp(width, 0, w_max)

            if width_offset is None:
                width_offset = w_o
            else:
                width_offset = N.clamp(width_offset, 0, w_max - 1)

            if height is None:
                height = h
            else:
                height = N.clamp(height, 0, h_max)

            if height_offset is None:
                height_offset = h_o
            else:
                height_offset = N.clamp(height_offset, 0, h_max - 1)

            self._cam['Width'].value = width
            self._cam['OffsetX'].value = width_offset
            self._cam['Height'].value = height
            self._cam['OffsetY'].value = height_offset

        width = self._cam['Width'].value
        width_offset = self._cam['OffsetX'].value
        height = self._cam['Height'].value
        height_offset = self._cam['OffsetY'].value

        new_vals = (width, width_offset, height, height_offset)

        if was_acquiring:
            self._cam.start_acquisition()

        return old_vals, new_vals

    def get_crop_meta_dict(self):
        """Returns current camera cropping as a dictionary.

        TODO remove? is this really needed that much?
        """

        w = self.width()
        wo = self.width_offset()
        h = self.height()
        ho = self.height_offset()
        return {'Width':w, 'OffsetX':wo, 'Height':h, 'OffsetY':ho}

    def _set_camera_feature(self, name, val):
        """Change camera settings.

        Note: OutOfRangeException related to features with certain increment
        (e.g. height, width) throws an exception which cannot be handle here.

        Parameters
        ----------
        name : string
            Name of the feature e.g. 'ExposureTime'
        val :
            Value to be set. Depending on the feature this might
            be int, string, bool etc.
        """

        if name in self._cam:
            cam_was_acquiring = self._cam.is_acquiring()
            try:
                # Try to set the value even if live feed is running.
                self._cam[name].value = val
                print(f"Feature '{name}' was succesfully set to {val}.")
            except AccessModeError:
                logging.warning(f"Could not change the feature {name} on the fly. Pausing and trying again.")
                self._cam.stop_acquisition()
                try:
                    self._cam[name].value = val
                    print(f"Feature '{name}' was succesfully set to {val}.")
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
                # Try to restart acquisition even if exceptions occurred.
                if cam_was_acquiring:
                    self._cam.start_acquisition()
        else:
            logging.warning(f"Feature '{name}' is not valid. Try again with valid name.")

    def save_camera_settings(self, relative_path):
        """Save camera settings into a file in given path."""

        if self._cam.is_initialized():
            abs_path = os.path.abspath(relative_path)
            logging.info(f"Trying to save camera settings to '{abs_path}'")
            try:
                self._cam.save_config_to_file(str(abs_path), overwrite=True)
            except:
                logging.error(f"Saving camera settings failed.")
                raise
        else:
            logging.warning(f"Camera not initialized. Cannot save camera settings.")

    def load_camera_settings(self, relative_path):
        """Load camera settings from a file in given path."""

        # Make into absolute path so that camazing gets it correctly.
        abs_path = os.path.abspath(relative_path)
        logging.info(f"Trying to load camera settings from '{abs_path}'")
        try:
            self._cam.load_config_from_file(str(abs_path))
        except:
            logging.error(f"Loading camera settings failed.")
            raise
