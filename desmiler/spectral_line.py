import math
import numpy as np

import curve_fit as cf

class SpectralLine:
    """ SpectralLine represents a spectral emission line on camera sensor. 

    The lines are expected to lie along y-axis in input data.

    Attributes
    ----------
        x : list
            List of x coordinates of data points forming the spectral line in xy-coordinates.
        y : list
            List of y coordinates of data points forming the spectral line in xy-coordinates.
        location : float
            Mean of x coordinates.
        circ_cntr_x : float
            Fitted circle center x-coordinate.
        circ_cntr_y : float
            Fitted circle center y-coordinate.
        circ_r : float 
            Fitted circle's radius.
        line_a : float
            a in line equation x = ay + b
        line_b : float
            b in line equation x = ay + b
        tilt_angle_degree_abs : float
            Absolute value of tilt angle in degrees. Measured as angle from line fit to vertical line.
    """

    def __init__(self, x, y):
        """ Initialize a SpectralLine object. 
        
        Parameters
        ----------
            x : list
                List of x coordinates of data points forming the spectral line in xy-coordinates.
            y : list
                List of y coordinates of data points forming the spectral line in xy-coordinates.    
        """

        self.x = x
        self.y = y
        self.location = np.mean(x)
        self.circ_cntr_x, self.circ_cntr_y, self.circ_r,_ = cf.LSF(x,y)
        self.line_a, self.line_b = cf.line_fit(x,y)
        self.tilt_angle_degree_abs = 90 - abs(math.atan(self.line_a) * 57.2957795)
