"""

This file contains all curve fitting used for the emission lines:
least squares circle fit (LSF), LMA circle fit, parabolic arc fit, and a LSF line fit.

"""


import scipy.stats as stats
import scipy.optimize as optimize
import scipy as sc
import numpy as np
import math

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
        return sc.sqrt((x-a)**2 + (y-b)**2)

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

def LMA(x,y):
    """Implements LMA algorithm from Chernov 2008 "Least squares fitting of circles and lines" 
    
    available at http://arxiv.org/abs/cs.CV/0301001.

    Parameters
    ----------
        x : list of x-coordinates
        y : list of y-coordinates
    
    Returns
    -------
        a : circle center x-coordinate
        b : circle center y-coordinate
        r : circle radius
        residu: Fitting residual. Not in use, always NaN.
    """

    def ui(x,y,theta):
        """u_i = x_i cos(theta) + y_i sin(theta) shorthand used in section 3.2."""
        u = x * math.cos(theta) + y * math.sin(theta)
        return u

    def zi(x,y):
        """z_i = x² + y² shorthand used in section 3.2."""
        z = x**2 + y**2
        return z

    def EE(A,D):
        """Returns E = sqrt(1+4AD): shorthand used in section 3.2. 
        (The paper has a typo in equation of P_i where factor 4 is missing.)
        """
        E = math.sqrt(1+4*A*D)
        return E

    def di(A,D,theta):
        """Distance from the circle to a point (x_i,y_i)
        
        d_i = 2 (P_i) / (1+sqrt(1+4AP_i)), [formula 2.8].
        """

        u = ui(x,y,theta)
        z = zi(x,y)
        E = EE(A,D)
        p = A*z + E*u + D
        ap = 1+4*A*p
        di = 2 * (p / (1+np.sqrt(ap)))
        return di

    def abr_to_adt(abr):
        """Convert natural circle parameters a,b, and r into LMA parameters A,D, and theta 
        used in the paper.

        Parameters
        ----------
            abr : array like, where
                abr[0] : circle center x-coordinate
                abr[1] : circle center y-coordinate
                abr[2] : circle radius
        
        Returns
        -------
            A : float
            D : float
            theta : float
        """

        a = abr[0]
        b = abr[1]
        r = abr[2]
        A = 1 / (2*r)
        B = - 2*A*a
        C = - 2*A*b
        D = (B*B + C*C - 1) / (4*A)
        """
        The paper was not too clear how to convert from B and C to theta (sec. 3.2).
        In PygMag library the conversion is implemented as theta = arccos(-a / np.sqrt(a*a + b*b)), 
        which produces the same result as using acos(B / (sqrt(1+4*A*D))). 

        Original definition for theta in the paper is:         

            B = sqrt(1+4AD) cos(theta), C = sqrt(1+4AD) sin(theta)
        """
        theta = np.arccos(-a / np.sqrt(a*a + b*b))
        return A,D,theta

    def f(abr):
        """Function to be minimized F = sum(d_i²)."""

        A,D,theta = abr_to_adt(abr)
        dist = di(A,D,theta)
        return dist*dist


    def jac(abr):
        """Jacobian of f as presented in the paper section 3.2."""

        A,D,theta = abr_to_adt(abr)
        u = ui(x,y,theta)
        z = zi(x,y)
        E = EE(A,D)
        p = A*z + E*u + D
        Qi = np.sqrt(1+4*A*p)
        dist = di(A,D,theta)
        Ri = (2*(1-A*dist/Qi))/(Qi+1)
        dA = (z + (2*D*u)/E) * Ri - (dist*dist) / Qi
        dD = (2*A*u / E + 1) * Ri
        dT = (-x * math.sin(theta) + y * math.cos(theta)) * E * Ri
        return np.array(list(zip(dA,dD,dT)))
    
    # Use LSF to get initial guess for circle parameters.
    a,b,r = LSF(x,y)

    # Minimize f with initial guess a,b,r. Uses Levenberg-Maquardt (method='lm')
    # as proposed in the paper.
    res = optimize.least_squares(f, (a,b,r), jac=jac, method='lm')
    return res.x[0], res.x[1], res.x[2], float('Nan')

def parabolicFit(x,y, p0=None):
    """ Fit a parabola to set of x,y points.

    Cannot be used replacing LSF and LMA as it does not produce 
    the same (a,b,r) parameter set as a result. Can be used to 
    compare original and desmiled spectral lines. Return parameters 
    a,b,c are as in sideways opening parabola equation x = ay^2 + by + c.
    """

    def parabola(x,a,b,c):
        return a*x**2 + b*x + c
    
    def jac(x,a,b,c):
        da = x**2
        db = x
        dc = np.ones_like(x)
        return np.array(list(zip(da,db,dc)))

    # Give coordinates in inverted order to get sideways parabola x = ay^2 + by + c
    p, pcov = sc.optimize.curve_fit(parabola, y, x, p0=p0, jac=jac)
    a = p[0]
    b = p[1]
    c = p[2]
    # The vertex
    # V = ((4*a*c - b**2) / (4*a), -b / (2*a))
    # Focus
    # F = ((4*a*c - b**2 + 1) / (4*a), -b / (2*a))
    return a,b,c

def line_fit(x,y):
    """Fit a least squares line to data.

    Inverted to fit almost vertical line as suggsted here:
    https://stats.stackexchange.com/questions/57685/line-of-best-fit-linear-regression-over-vertical-line

    Parameters
    ----------
        x : list 
            List of x coordinates
        x : list 
            List of u coordinates

    Returns
    -------
        a : float
            a in line equation x = ay + b
        b : float
            b in line equation x = ay + b
    """
    A,B,_,_,_ = stats.linregress(y, x)
    return 1/A,-B/A

if __name__ == '__main__':
    print("curve_fit.py called as script. No need to run anything.")