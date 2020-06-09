import scipy.optimize as optimize
import numpy as np

def LSF(x,y):
    """Fit a least squares error circle into data points.

    Adapted from:
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    Parameters
    ----------
        x : list of x-coordinates
        y : list of y-coordinates
    
    Returns
    -------
        a : circle center x-coordinate
        b : circle center y-coordinate
        r : circle radius
        residu: Fitting residual: sum of squared differences.  
    """

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    def di(a, b):
        """ Calculate the distance of each 2D points from the center (a, b) """
        return np.sqrt((x-a)**2 + (y-b)**2)

    def f(c):
        """ Calculate the algebraic distance between the data points and the 
            mean circle centered at c=(a, b) 
        """
        Ri = di(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate)

    a, b = center
    Ri     = di(*center)
    R      = Ri.mean()
    residu = sum((Ri - R)**2)

    return a,b,R,residu