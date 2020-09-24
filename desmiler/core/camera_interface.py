"""

Provides camera interface to be used elswhere in the program.
Should be easier to swap to a different backend if desired.

This is very simplistic usage of camazing package. We encourage you
to play around and explore its capabilities.

"""

class CameraInterface:

    """
    TODO should do a setting dump and load possibility for scanning sessions?

    """

    def __init__(self):
        print(f"Initializing camera interface")
        # TODO load camazing camera

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
