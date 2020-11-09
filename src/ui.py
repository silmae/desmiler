"""

Console UI for controlling various funktionalities.

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

        if self.sc is not None:
            print(f"Closing existing session '{self.sc.session_name}' before opening a new one.")
            self.close_session()

        self.sc = ScanningSession(session_name)
        print(f"Session '{self.sc.session_name}' ready.")

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

    def close_preview(self):
        if self.preview is not None:
            del self.preview

    def start_preview(self):
        """Start a new preview for inspecting the camera feed."""

        self.preview = Preview()
        self.preview.start()

    def shoot_dark(self):
        """Shoots and saves a dark frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
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
        """Shoots and saves a white frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
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
        """Shoots and saves a peaky light frame.

        If a session exists, dark should be saved to its folder, otherwise
        to preview folder.
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

    def crop(self, width=None, width_offset=None, height=None, height_offset=None, full=False):
        # TODO this might be unnecessary as the idea is to control everything with control files. Remove?
        if self.sc is not None:
            self.sc.crop(width, width_offset, height, height_offset, full)
        else:
            print(f"Asked for cropping but there is no active session to crop.")

    def run_scan(self):
        if self.preview is not None:
            preview_was_running = self.preview.is_running
            self.preview.stop()

        if self.sc is not None:
            self.sc.run_scan()
        else:
            print(f"Asked to run a scan but there is no active session to run.")

        if self.preview is not None and preview_was_running:
            self.preview.start()

    # def inspect_light(self):
    #     if self.sc is not None:
    #         path = self.sc.session_root + '/' + P.ref_light_name
    #         frame_ds = F.load_frame(path)
    #         frame_inspector.plot_frame(frame_ds)
    #         frame_inspector.plot_frame_spectra(frame_ds)

            # f_vals = frame.values
            # log_frame = frame
            # log_frame.values = np.log(f_vals)
            # frame_inspector.plot_frame(frame)
            # frame_inspector.plot_frame_spectra(log_frame)

    def generate_examples(self):
        synthetic.generate_all_examples()

    def show_examples(self):
        synthetic.show_frame_examples()
        synthetic.show_cube_examples()

    def make_reflectance_cube(self):
        if self.sc is not None:
            self.sc.make_reflectance_cube()
        else:
            logging.warning(f"Kääk")

    def make_desmiled_cube(self):
        if self.sc is not None:
            self.sc.desmile_cube(shift_method=0)
            self.sc.desmile_cube(shift_method=1)
        else:
            logging.warning(f"Kääk")

    def show_cube(self):
        if self.sc is not None:
            self.sc.show_cube()
        else:
            logging.warning(f"Kääk")

    def show_shift(self):
        if self.sc is not None:
            self.sc.show_shift()
        else:
            logging.warning(f"Kääk")

    def show_light(self):
        if self.sc is not None:
            self.sc.show_light()
        else:
            logging.warning(f"Kääk")

    def exposure(self, value=None) -> int:
        if self.sc._cami is not None:
            return self.sc.exposure(value)
        elif self.preview is not None:
            return self.preview._cami.exposure(value)
        else:
            logging.warning(f"Cannot set exposure as no Session or Preview exists.")

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    ui = UI()
    # ui.start_session(P.example_scan_name)
    # ui.inspect_light()
    # sc = ui.sc

    # raw = F.load_cube(os.path.abspath(P.path_rel_scan + P.example_scan_name + '/' + P.cube_reflectance_name + '.nc'))
    # desmiled = F.load_cube(os.path.abspath(P.path_rel_scan + P.example_scan_name + '/' + P.cube_desmiled_lut + '.nc'))
    # ci = CubeInspector(raw, desmiled, desmiled, 'reflectance')
    # ci.show()

    # ui.run_scan()
    # ui.crop(200,100,150,300)
    # ui.shoot_dark()x
    # ui.shoot_white()
    # ui.shoot_light()

    # ui.start_freeform_session()
    # ui.show_cube()
    # ui.run_scan()
    # ui.inspect_light()
    ui.start_session('samir')
    # ui.shoot_light()
    # ui.shoot_dark()
    # ui.shoot_white()
    # ui.make_reflectance_cube()
    # ui.show_shift()
    # ui.show_light()
    # ui.make_desmiled_cube()
    # ui.run_scan()

    ui.show_cube()
    # ui.show_shift()

    # cami = CameraInterface()
    # d = cami._cam.features()

    # for i in range(len(d)):
    #     print(d[i])
    # print(f"AcquisitionFrameRateAuto : {cami._cam['AcquisitionFrameRateAuto'].value}")
    # print(f"SingleFrameAcquisitionMode : {cami._cam['SingleFrameAcquisitionMode'].info()}")
    # print(f"SingleFrameAcquisitionMode : {cami._cam['SingleFrameAcquisitionMode'].info()}")

    # ui.start_preview()
    # ui.close_preview()
    # ui.shoot_dark()


