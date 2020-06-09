import scipy.optimize as optimize
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
        return math.sqrt((x-a)**2 + (y-b)**2)

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
    """Implements LMA algorithm from Chernov 2008.

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

    def abrToADT(abr):
        """Convert general circle equation parameters a,b, and r 
        (circle center and radius) into parameters A,D, and theta 
        used in the paper.

        Parameters
        ----------
        abr : array like, where
            abr[0] : circle center x-coordinate
            abr[1] : circle center y-coordinate
            abr[2] : circle radius
        """

        a = abr[0]
        b = abr[1]
        r = abr[2]
        A = 1 / (2*r)
        B = - 2*A*a
        C = - 2*A*b
        D = (B*B + C*C - 1) / (4*A)
        # Not sure how the theta should be calculated. 
        # These are derived from the two equations given when theta is defined:
        # B = sqrt(1+4AD) cos(theta), C = sqrt(1+4AD) sin(theta)
        #theta = math.acos(B / (math.sqrt(1+4*A*D))) # from B
        #theta = math.asin(C / (math.sqrt(1+4*A*D))) # from C
        # The following form for calculating theta is used in PygMag library, 
        # so it might be correct. It gives the same result as from B.
        theta = np.arccos(-a / np.sqrt(a*a + b*b))
        return A,D,theta

    def f(abr):
        """Function to be minimized F = sum(d_i²)."""

        A,D,theta = abrToADT(abr)
        dist = di(A,D,theta)
        return dist*dist


    def jac(abr):
        """Jacobian of f as presented in the paper section 3.2."""

        A,D,theta = abrToADT(abr)
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