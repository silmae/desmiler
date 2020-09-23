"""

Some control over sessions to create metadata and
correct folder structure.

"""

import os

import properties as P
import file_handling as F

class ScanningSession:

    # Session log? At least some metadata (datetime, camera settings etc.) should be saved I think.

    def __init__(self, session_name:str):
        print(f"Creating session '{session_name}'")
        self.session_root = P.scan_folder_name + '/' + session_name
        self.session_name = session_name

        if self.session_exists():
            print(f"Found existing session. Loading files..")
            print(f"And now I pretend to load those files into memory, but not really doing nothing.")
        else:
            F.create_directory()

    def session_exists(self) -> bool:
        if os.path.exists(self.session_root):
            return True
        else:
            return False



