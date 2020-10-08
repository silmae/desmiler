"""

Console UI for controlling various funktionalities.

"""


from imaging.scanning_session import ScanningSession
import imaging.scanning_session as scanning_session
from core.camera_interface import CameraInterface
import analysis.frame_inspector as frame_inspector
from core import properties as P
from utilities import file_handling as F
from imaging.preview import Preview

import numpy as np

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

    def __del__(self):
        if self.sc is not None:
            self.sc.close()
            del self.sc
        if self.preview is not None:
            self.preview.close()
            del self.preview

    def start_session(self, session_name:str) -> ScanningSession:
        """Start new named scanning session with its own folder."""

        print(f"Trying to start new session '{session_name}'")
        if self.sc is not None:
            print(f"Found existing session '{self.sc.session_name}'. Cannot create a new one before closing.")
            self.close_session()

        self.sc = ScanningSession(session_name)
        print(f"Created new scanning session '{self.sc.session_name}'.")

    # TODO start_session_copy() copy an existing session as a new session

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

        # TODO forcibly show the result with circle and line plots

        if self.sc is not None:
            self.sc.shoot_light()
        else:
            print(f"No session running. Start a session before shooting light frame.")

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        # TODO this might be unnecessary as the idea is to control everything with control files. Remove?
        if self.sc is not None:
            self.sc.crop(width, width_offset, height, height_offset, full)
        else:
            print(f"Asked for cropping but there is no active session to crop.")

    def run_scan(self):
        if self.sc is not None:
            self.sc.run_scan()
        else:
            print(f"Asked to run a scan but there is no active session to run.")

    def inspect_light(self):
        if self.sc is not None:
            path = self.sc.session_root + '/' + P.ref_light_name
            frame_ds = F.load_frame(path)
            frame = frame_ds.frame
            frame_inspector.plot_frame(frame)
            frame_inspector.plot_frame_spectra(frame)

            # f_vals = frame.values
            # log_frame = frame
            # log_frame.values = np.log(f_vals)
            # frame_inspector.plot_frame(frame)
            # frame_inspector.plot_frame_spectra(log_frame)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    ui = UI()
    # ui.start_session('my_new_session')
    # ui.run_scan()
    # ui.crop(200,100,150,300)
    # ui.shoot_dark()x
    # ui.shoot_white()
    # ui.shoot_light()

    ui.start_freeform_session()
    # ui.inspect_light()

    # ui.start_preview()

