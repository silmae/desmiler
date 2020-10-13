import math
import numpy as np

from core import curve_fit as cf


class SpectralLine:
    """ SpectralLine object represents a spectral emission line on camera sensor. 

    The lines are expected to lie along y-axis in input data.

    Fits a circle and a line to data points. Circle is defined by (x-a)^2 + (y-b)^2 = r^2,
    where (a,b) is its center point and r its radius. 

    Attributes
    ----------
        x : list
            List of x coordinates of data points forming the spectral line in xy-coordinates.
        y : list
            List of y coordinates of data points forming the spectral line in xy-coordinates.
        location : float
            Location of the SL on x-axis. 
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
        # Correct location of the SL assumed mean of points. Can also use 
        # circ_cntr_x - circ_r. Using mean prevents the lines moving too far from 
        # their original positions.
        self.location = np.mean(x)
        # LSF can be changed to LMA if desired. 
        # But not for parabolic as they have different return values!
        self.circ_cntr_x, self.circ_cntr_y, self.circ_r,_ = cf.LSF(x,y)
        self.line_a, self.line_b = cf.line_fit(x,y)
        self.tilt_angle_degree_abs = 90 - abs(math.atan(self.line_a) * 57.2957795)
        a_tan = math.degrees(-math.atan(self.line_a))
        self.tilt = math.copysign(90 - abs(a_tan), a_tan)

        # Parabolic fit to estimate curvature
        p0 = [1 / (2 * self.circ_r), 0.0, self.location]
        a, _, _ = cf.parabolicFit(x,y,p0)
        self.curvature = 2 * a
