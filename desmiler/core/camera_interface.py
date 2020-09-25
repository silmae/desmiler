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

from camazing import CameraList
from core import properties as P

class CameraInterface:

    """
    TODO should do a setting dump and load possibility for scanning sessions?

    """

    # Should not be used directly unless it can't be avoided.
    _cam = None

    def __init__(self):
        print(f"Initializing camera interface")
        # TODO load camazing camera
        self._initCam()

    def turn_on(self):
        # TODO turn the camera on
        print("Pretending to turn camera on.")

    def turn_off(self):
        # TODO turn the camera off
        print("Pretending to turn camera off.")

    def get_frame(self):
        print("Would get you a frame if I know how.")

    def exposure(self, value=None):
        """Set or print exposure"""
        if value is None:
            print(f"Now I would print the value of 'exposure time'")
        else:
            print(f"Now I would set the value of 'exposure time'")

    def gain(self, value=None):
        """Set or print gain"""
        if value is None:
            print(f"Now I would print the value of 'gain'")
        else:
            print(f"Now I would set the value of 'gain'")

    def width(self, value=None):
        """Set or print width"""
        if value is None:
            print(f"Now I would print the value of 'width'")
        else:
            print(f"Now I would set the value of 'width'")

    def width_offset(self, value=None):
        """Set or print width_offset"""
        if value is None:
            print(f"Now I would print the value of 'width_offset'")
        else:
            print(f"Now I would set the value of 'width_offset'")

    def height(self, value=None):
        """Set or print height"""
        if value is None:
            print(f"Now I would print the value of 'height'")
        else:
            print(f"Now I would set the value of 'height'")

    def height_offset(self, value=None):
        """Set or print height_offset"""
        if value is None:
            print(f"Now I would print the value of 'height_offset'")
        else:
            print(f"Now I would set the value of 'height_offset'")

    def _initCam(self):
        """Initializes the camera.

        Assumes only one camera is connected to the machine so the first one
        in camazing's CameraList will be used.

        Camera settings are loaded from 'settings.toml' which should be placed in the
        same folder with code files.

        Raises
        ------
        Raises some misc exceptions if camera initialization fails foca some reason.
        Should look more deeply into camazing to find out all that can go wrong.
        """

        cameras = CameraList('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

        if len(cameras) < 1:
            raise RuntimeError("Could not find the camera. Camera not initialized.")

        self._cam = cameras[0]

        try:
            print("Initializing camera...", end=' ')
            self._cam.initialize()
            print("done")
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
