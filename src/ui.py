"""

Console UI for controlling various funktionalities.

"""


from imaging.scanning_session import ScanningSession
import imaging.scanning_session as scanning_session
from core.camera_interface import CameraInterface
from core import properties as P
from utilities import file_handling as F
from imaging.preview import Preview

import logging

class UI:

    sc = None
    preview = None

    def __init__(self):
        print("Initializing the UI.")
        F.create_default_directories()
        print("I cannot yet do anything but next, you might like to start "
              "a named scanning session 'start_session()' for later analysis, "
              "a freeform session to make quick experiments 'start_freeform_session()' or "
              "a preview to see camera feed 'start_preview()'.")

        scanning_session.create_example_scan()

    def start_session(self, session_name:str) -> ScanningSession:
        """Start new named scanning session with its own folder."""

        print(f"Trying to start new session '{session_name}'")
        if self.sc is not None:
            print(f"Found existing session '{self.sc.session_name}'. Cannot create a new one before closing.")
            self.close_session()

        self.sc = ScanningSession(session_name)
        print(f"Created new scanning session '{self.sc.session_name}'.")

    def start_freeform_session(self):
        """Freeform session will always overwrite old content with the same name."""

        self.start_session(P.freeform_session_name)

    def close_session(self):
        print(f"Closing session '{self.sc.session_name}'")
        # TODO Check that everything is OK before closing
        self.sc.close()
        del self.sc

        print(f"Session closed.")

    def start_preview(self):
        """Start a new preview for inspecting the camera feed.

        Closes existing session as they use the same camera resource.
        """

        print(f"Trying to start preview")
        if self.sc is not None:
            print(f"An existing session must be closed before starting the preview.")
            self.close_session()

        self.preview = Preview()
        self.preview.start()


    def shoot_dark(self):
        """Shoots and saves a dark frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
        """

        if self.sc is not None:
            self.sc.shoot_dark()
        else:
            print(f"No session running. Start a session before shooting dark frame.")

    def shoot_white(self):
        """Shoots and saves a white frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
        """

        if self.sc is not None:
            self.sc.shoot_white()
        else:
            print(f"No session running. Start a session before shooting white frame.")

    def shoot_light(self):
        """Shoots and saves a peaky light frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
        """

        if self.sc is not None:
            self.sc.shoot_light()
        else:
            print(f"No session running. Start a session before shooting light frame.")

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        if self.sc is not None:
            self.sc.crop(width, width_offset, height, height_offset, full)
        else:
            print(f"Asked for cropping but there is no active session to crop.")

    def run_scan(self):
        if self.sc is not None:
            self.sc.run_scan()
        else:
            print(f"Asked to run a scan but there is no active session to run.")


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    ui = UI()
    ui.start_session('my_new_session')
    ui.run_scan()
    ui.crop(200,100,150,300)
    ui.shoot_dark()
    ui.shoot_white()
    ui.shoot_light()
    ui.start_freeform_session()
    ui.start_preview()

