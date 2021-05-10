"""

Console user interface intended to be used with ipython. The main at the
end of the file should contain at least the line "ui = UI()", so that
running magic command "%run ui.py" in ipython console gives a ready-to-use
ui object.

Most of the commands expect that there exists a scanning session to connect the data to.
Scanning sessions are bound to folder "/scans/<scan_name>".

"""
from core.camera_interface  import CameraInterface

from imaging.scanning_session import ScanningSession
from imaging import scanning_session as scanning_session
import analysis.frame_inspector as frame_inspector
from core import properties as P
from utilities import file_handling as F
from imaging.preview import Preview

import synthetic_data as synthetic

import logging

class UI:

    sc = None
    preview = None

    def __init__(self):
        print("Initializing the UI.")
        F.create_default_directories()
        print("Welcome to desmiler UI! \n")
        print("You can start by generating examples by running ui.generate_examples().")
        print("They can be viewed by running ui.show_examples().")
        print("If you have a connected camera, you may want to run ui.start_preview() to show "
              "raw video feed from the camera.")
        print("Run ui.start_session(<session_name>) to start a new session or to open an existing one.")
        print("If the session contains an existing hyperspectral image cube, you can inspect it by calling "
              "ui.show_cube().")

        scanning_session.create_example_scan()

    def __del__(self):
        # FIXME they are both trying to delete the camera, which will throw an error
        if self.sc is not None:
            self.sc.close()
            del self.sc
        if self.preview is not None:
            self.preview.close()
            del self.preview

    def start_session(self, session_name:str):
        """Start new named scanning session with its own folder."""

        if self.sc is not None:
            print(f"Closing existing session '{self.sc.session_name}' before opening a new one.")
            self.close_session()

        self.sc = ScanningSession(session_name)
        print(f"Session '{self.sc.session_name}' ready.")

    def start_freeform_session(self):
        """Freeform session will always overwrite old content with the same name."""

        self.start_session(P.freeform_session_name)

    def close_session(self):
        """Closes existing session. """

        print(f"Closing session '{self.sc.session_name}'")
        self.sc.close()
        del self.sc

        print(f"Session closed.")

    def close_preview(self):
        """Closes existing preview."""

        if self.preview is not None:
            del self.preview

    def start_preview(self):
        """Starts a new preview for inspecting the camera feed."""

        camera_settings_path = None
        if self.sc:
            camera_settings_path = self.sc.camera_setting_path
        self.preview = Preview(camera_settings_path)
        self.preview.start()

    def shoot_dark(self):
        """Shoots and saves a dark frame for current session.

        Pauses a preview if one is running.
        """

        if self.preview is not None:
            preview_was_running = self.preview.is_running
            self.preview.stop()

        if self.sc is not None:
            self.sc.shoot_dark()
        else:
            print(f"No session running. Start a session before shooting dark frame.")

        if self.preview is not None and preview_was_running:
            self.preview.start()

    def shoot_white(self):
        """Shoots and saves a white frame for current session.

        Pauses a preview if one is running.
        """
        if self.preview is not None:
            preview_was_running = self.preview.is_running
            self.preview.stop()

        if self.sc is not None:
            self.sc.shoot_white()
        else:
            print(f"No session running. Start a session before shooting white frame.")

        if self.preview is not None and preview_was_running:
            self.preview.start()

    def shoot_light(self):
        """Shoots and saves a peaky light frame for current session.

        Pauses a preview if one is running.
        """

        if self.preview is not None:
            preview_was_running = self.preview.is_running
            self.preview.stop()

        if self.sc is not None:
            self.sc.shoot_light()
        else:
            print(f"No session running. Start a session before shooting light frame.")

        if self.preview is not None and preview_was_running:
            self.preview.start()

    def run_scan(self):
        """Run a scan using active session's control file for scanning parameters."""

        if self.preview is not None:
            preview_was_running = self.preview.is_running
            self.preview.stop()

        if self.sc is not None:
            self.sc.run_scan()
        else:
            print(f"Asked to run a scan but there is no active session to run.")

        if self.preview is not None and preview_was_running:
            self.preview.start()

    def generate_examples(self):
        """Generates all available examples. This takes quite a long time.

        You can use "synthetic_data.py"'s own methods to generate just some of the examples.
        """

        synthetic.generate_all_examples()

    def show_examples(self):
        """Shows all available examples. """

        synthetic.show_frame_examples()
        synthetic.show_cube_examples()

    def make_reflectance_cube(self):
        """Make a reflectance cube using existing raw cube, dark reference, and white reference."""

        if self.sc is not None:
            self.sc.make_reflectance_cube()
        else:
            logging.warning(f"No active scanning session exists. Cannot create reflectance cube.")

    def make_desmiled_cube(self):
        """Run smile correction for a reflectance cube with interpolated and lookup table shifts."""

        if self.sc is not None:
            self.sc.desmile_cube(shift_method=0)
            self.sc.desmile_cube(shift_method=1)
        else:
            logging.warning(f"No active scanning session exists. Cannot desmile a cube.")

    def show_cube(self):
        """Start the CubeInspector."""

        if self.sc is not None:
            self.sc.show_cube()
        else:
            logging.warning(f"No active scanning session exists. Cannot show a cube.")

    def show_shift(self):
        """Show the shift matrix. Use this to check that the matrix looks reasonable."""

        if self.sc is not None:
            self.sc.show_shift()
        else:
            logging.warning(f"No active scanning session exists. Cannot show the shift matrix.")

    def show_light(self):
        """Show light reference frame to see how well the arc fits fit."""

        if self.sc is not None:
            self.sc.show_light()
        else:
            logging.warning(f"No active scanning session exists. Cannot show light reference.")

    def exposure(self, value=None) -> int:
        """Set or show the current exposure in microseconds. Useful for preview.

        The run_scan() method does not respect this exposure, but the one set in the control file.
        """

        if self.sc._cami is not None:
            return self.sc.exposure(value)
        elif self.preview is not None:
            return self.preview._cami.exposure(value)
        else:
            logging.warning(f"Cannot set exposure as no Session camera or Preview exists.")

if __name__ == '__main__':
    # set logging parameters here
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    # create an UI object to be used from the ipython console
    ui = UI()
    # ui.start_freeform_session()
    # ui.start_preview()
    ui.start_session('window1')
    # ui.run_scan()
    # ui.start_preview()
    # ui.make_desmiled_cube()
    # ui.show_cube()