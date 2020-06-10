import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as signal
import xarray as xr
import math

from spectral_line import SpectralLine

def construct_bandpass_filter(peak_light_frame, location_estimates, filter_window_width):
    """ Constructs a bandpass filter.

    Generated filters may be narrower than given filter_window_width if locations 
    are packed too densly together.

    Parameters
    ----------
        peak_light_frame : xarray DataArray
            Frame with spectral lines as DataArray. Expected dimension names 
            of the array are x and y. Spectral lines are expected to lie 
            along y-dimension.
        location_estimates : list
            User defined estimates of x-location where to find a spectral line. 
            The filter is constructed around these locations. Locations should 
            be further that filter_window_width away from each other.
        filter_window_width : int
            How wide filter windows are generated around locations. 

    Returns
    -------
        low : numpy array
            One-dimensional lower limit column-wise filter (same size as x). Filled with zeros.
        high : numpy array
            One-dimensional higher limit column-wise filter (same size as x). Filled with zeros.
    """

    # Initialize return values.
    low = np.zeros(peak_light_frame.x.size)
    high = np.zeros(peak_light_frame.x.size)
    # Half width for calculations.
    w = int(filter_window_width / 2)
    # x-coordinate of previous filter window end point.
    last_stop = 0

    for _,le in enumerate(location_estimates):
        try:
            max_val = peak_light_frame.isel(x=slice(le-w,le+w)).max(dim=['x','y'])
        except ValueError as ve:
            # Use maximum of the whole frame as a backup.
            max_val = np.max(peak_light_frame.values)
            
        start = le-w
        # Ensure windows are separate even if given locations are closer than w. 
        if start <= last_stop:
            start = last_stop + 1
        # May result in narrower filter but does not push following filters away from their intended position.
        stop  = le + w
        # Check array bounds
        if stop >= len(high):
            stop = len(high)-1
        last_stop = stop

        # Fill the area around spectral line locations with non-zero values.
        idx = np.arange(start=start,stop=stop)
        for _,j in enumerate(idx):
            high[j] = max_val * 1.2

    return low,high

def construct_spectral_lines(peak_light_frame, location_estimates, bandpass):
    """ Constructs a spectral lines found from given frame. 

    Spectral lines are expected to be found from location_estimates, which should be 
    the same that is provided for construct_bandpass_filter() method.

    Parameters
    ----------
        peak_light_frame
            Frame with spectral lines as DataArray. Expected dimension names 
            of the array are x and y. Spectral lines are expected to lie 
            along y-dimension.
        location_estimates
            User defined estimates of x-location where to find a spectral line. 
            The filter is constructed around these locations. Locations should 
            be further that filter_window_width away from each other.
        bandpass : (array-like, array-like)
            A bandpass filter as provided by construct_bandpass_filter() method.
    
    Returns
    -------
        spectral_line_list
            A list of SpectralLine objects.
        count_rejected
            How many rows were rejected because of wrong peak count.

    """

    # Hard coded peak height for the peak finding algorithm.
    peakWidth = 3
    
    rowList = []
    accepted_row_index = []
    spectral_line_list = []
    count_rejected = 0

    # Iterate frame rows to find peaks from given locations on each row.
    for i in range(peak_light_frame.y.size):

        row = peak_light_frame.isel(y=i).values
        rowPeaks, _ = signal.find_peaks(row, height=bandpass, width=peakWidth)

        if len(rowPeaks) == len(location_estimates):
            accepted_row_index.append(i)
            rowList.append(rowPeaks)
        else:
            count_rejected += 1

    # Once each row that succesfully found same amount of peaks that there are location_estimates,
    # we can form the actual spectral line objects.
    for i,rowPeaks in enumerate(rowList):
            line = SpectralLine(rowPeaks,accepted_row_index) 
            # Discard lines with too small radius. They are false alarms.
            if line.r > peak_light_frame.x.size:
                spectral_line_list.append(line)

    if len(spectral_line_list) < 1:
        raise RuntimeWarning(f"All spectral lines were ill formed.")

    return spectral_line_list, count_rejected


def construct_shift_matrix(spectral_lines, w):
    """Constructs a desmiling shift distance matrix.

    Parameters
    ----------

    spectralLines : List<SpectralLine>
        A list of spectral lines to base the desmiling on as returned by 
        FrameTools.getSpectralLines().
    w: int
        Width of the frame to be desmiled.
    h: int
        Height of the frame to be desmiled.
    
    Returns
    -------
    shift_matrix : Xarray DataArray
        Desmile distance matrix. Use FrameTools.getDesmileIndexArrays() to 
        get new indices.

    Raises
    ------
    TypeError 
        If given frame or spectralLines was None.
    RuntimeError
        If given spectralLines was empty.
    """

    shift_matrix = xr.DataArray(np.zeros((h,w)), dims=('y','x'))

    # Switch to single circle method if only one spectral line was recieved.
    if len(spectral_lines) == 1:
        shift_matrix = _single_circle_shift(shift_matrix, spectral_lines, w)
    else:
        print("Desmiling with Interpolated Circles method.")
        shift_matrix = _multi_circle_shift(shift_matrix, spectral_lines, w)

    return shift_matrix

def _single_circle_shift(shift_matrix, spectral_lines, w):

    sl = spectral_lines[0]
    for x in range(shift_matrix.y.size):
        xx = x - sl.circ_cntr_y
        theta = math.asin(xx / sl.circ_r)
        py = (1 - math.cos(theta)) * math.copysign(sl.circ_r, sl.circ_cntr_x)
        for l in range(w):
            shift_matrix.values[x, l] = py

    return shift_matrix

def _multi_circle_shift(shift_matrix, spectral_lines, w):
        
    # x coordinates of spectral lines. First element set to 0, last to the width of the frame.
    x_coords = []

    for i,sl in enumerate(spectral_lines):
        pl = sl.location
        x_coords.append(pl)

        if i == 0 or i == (len(spectral_lines)-1):
        # Add an element to beginning and end of list        
            x_coords.append(pl)

    # Overwrite the extra elements
    x_coords[0] = 0
    x_coords[len(x_coords)-1] = w

    for row_idx in range(shift_matrix.y.size):
        shifts = []
        for i,sl in enumerate(spectral_lines):
            h = row_idx - sl.circ_cntr_y
            theta = math.asin(h / sl.circ_r)
            d = (1 - math.cos(theta)) * math.copysign(sl.circ_r, sl.circ_cntr_x)
            shifts.append(d)
            if i == 0 or i == (len(spectral_lines)-1):
            # Set first element same as the second, and last same as second to last.
                shifts.append(d)

        f = interp1d(x_coords, shifts)

        row = np.arange(w)
        shift_linear_fit = f(row)

        for l,d in enumerate(shift_linear_fit):
            shift_matrix.values[x,l] = d

    return shift_matrix