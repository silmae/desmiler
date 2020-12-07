"""

This file contains the core functionality of the smile correction process, i.e.,
bandpass filter construction, spectral line construction, shift matrix construction and
shift application.

"""

import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as signal
import xarray as xr
import math

from core.spectral_line import SpectralLine
from core import properties as P

def construct_bandpass_filter(peak_light_frame, location_estimates, filter_window_width):
    """ Constructs a bandpass filter for given frame.

    Generated filter windows may be narrower than given filter_window_width if locations 
    are packed too densly together.

    Parameters
    ----------
        peak_light_frame : xarray Dataset
            Frame with spectral lines as Dataset. Spectral lines are expected to lie
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
            One-dimensional lower limit column-wise filter (same size as peak_light_frame.x). Filled with zeros.
        high : numpy array
            One-dimensional higher limit column-wise filter (same size as peak_light_frame.x).
    """

    # Initialize return values (low and high value vectors are both the width of the frame).
    low = np.zeros(peak_light_frame[P.dim_x].size)
    high = np.zeros(peak_light_frame[P.dim_x].size)
    # Half width for calculations.
    w = int(filter_window_width / 2)
    # x-coordinate of previous filter window end point.
    last_stop = 0

    for _,le in enumerate(location_estimates):
        try:
            max_val = peak_light_frame.isel(x=slice(le-w,le+w)).max(dim=[P.dim_x,P.dim_x])
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

def construct_spectral_lines(peak_light_frame, location_estimates, bandpass, peak_width=3):
    """ Constructs spectral lines found from given frame. 

    Spectral lines are expected to be found from location_estimates, which should be 
    the same that is provided for construct_bandpass_filter() method.

    Parameters
    ----------
        peak_light_frame : xarray Dataset
            Frame with spectral lines as Dataset. Expected dimension names 
            of the array are x and y. Spectral lines are expected to lie 
            along y-dimension.
        location_estimates : list int
            User defined estimates of x-location where to find a spectral line. 
            The filter is constructed around these locations. Locations should 
            be further that filter_window_width away from each other.
        bandpass : (array-like, array-like)
            A bandpass filter as provided by construct_bandpass_filter() method.
        peak_width
            Peak width passed to signal.find_peaks()
    
    Returns
    -------
        spectral_line_list : list SpectralLine
            A list of SpectralLine objects.

    """

    rowList = []
    accepted_row_index = []
    spectral_line_list = []

    # Iterate frame rows to find peaks from given locations on each row.
    for i in range(peak_light_frame.y.size):

        row = peak_light_frame.isel(y=i).values
        rowPeaks, _ = signal.find_peaks(row, height=bandpass, width=peak_width)

        if len(rowPeaks) == len(location_estimates):
            accepted_row_index.append(i)
            rowList.append(rowPeaks)

    rowList = np.asarray(rowList)
    accepted_row_index = np.asarray(accepted_row_index)

    # Once each row that succesfully found same amount of peaks that there are location_estimates,
    # we can form the actual spectral line objects.
    for i in range(len(rowList[0])):
            x = rowList[:,i]
            y = accepted_row_index
            line = SpectralLine(x,y) 
            # Discard lines with too small radius. They are false alarms.
            if line.circ_r > peak_light_frame.x.size:
                spectral_line_list.append(line)

    if len(spectral_line_list) < 1:
        raise RuntimeWarning(f"All spectral lines were ill formed.")

    return spectral_line_list


def construct_shift_matrix(spectral_lines, w, h):
    """Constructs a shift (distance) matrix for smile correction.

    Parameters
    ----------

    spectral_lines : list SpectralLine
        A list of spectral lines to base the desmiling on.
        Use construct_spectral_lines() to acquire them.
    w: int
        Width of the frame to be desmiled.
    h: int
        Height of the frame to be desmiled.
    
    Returns
    -------
    shift_matrix : xarray DataArray
        Shift distance matrix. Use _shift_matrix_to_index_matrix() to 
        get new indices.
    """

    shift_matrix = xr.DataArray(np.zeros((h,w)), dims=('y','x'))

    # Switch to single circle method if only one spectral line was recieved.
    if len(spectral_lines) == 1:
        shift_matrix = _single_circle_shift(shift_matrix, spectral_lines, w)
    else:
        shift_matrix = _multi_circle_shift(shift_matrix, spectral_lines, w)

    return shift_matrix

def _single_circle_shift(shift_matrix, spectral_lines, w):
    """ Create shifts using a single spectral line. """

    sl = spectral_lines[0]
    for x in range(shift_matrix.y.size):
        xx = x - sl.circ_cntr_y
        theta = math.asin(xx / sl.circ_r)
        py = (1 - math.cos(theta)) * math.copysign(sl.circ_r, sl.circ_cntr_x)
        for l in range(w):
            shift_matrix.values[x, l] = py

    return shift_matrix

def _multi_circle_shift(shift_matrix, spectral_lines, w):
    """ Create shift matrix by interpolating several spectral lines. """
        
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
            shift_matrix.values[row_idx,l] = d

    return shift_matrix

def apply_shift_matrix(target, shift_matrix, method=0, target_is_cube=True):
    """ Apply shift matrix to a hyperspectral image cube or a single frame. 

    Parameters
    ----------
        target : xarray Dataset
            Target cube or frame, specify with target_is_cube parameter.
        shift_matrix
            The shift matrix to apply as given by construct_shift_matrix().
        method
            Either 0 for lookup table method or 1 for row interpolation method.
            Interpolation is slower but more accurate.
    
    Returns
    -------
        xarray Dataset
            Desmiled target as a dataset. 

    Raises
    ------
        Value error if method other than 0 or 1.
    """

    if method == 0:
        if target_is_cube:
            desmiled_target = _lut_shift_cube(target, shift_matrix)
        else:
            desmiled_target = _lut_shift_frame(target, shift_matrix)
    elif method == 1:
        if target_is_cube:
            desmiled_target = _intr_shift_cube(target, shift_matrix)
        else:
            desmiled_target = _intr_shift_frame(target, shift_matrix)
    else:
        raise ValueError(f"Method must be either 0 or 1. Was {method}.")

    return desmiled_target

def _lut_shift_cube(cube, shift_matrix):
    """ Apply lookup table shift for a hyperspectral image cube. """

    ix,iy  = _shift_matrix_to_index_matrix(shift_matrix)
    vals = np.zeros_like(cube.reflectance)
    for i,frame in enumerate(cube.reflectance.values):        
        vals[i,:,:] = frame[iy, ix]
    cube.reflectance.values = vals
    return cube

def _lut_shift_frame(frame, shift_matrix):
    """ Apply lookup table shift for a single frame. """

    ix, iy = _shift_matrix_to_index_matrix(shift_matrix)
    frame.values[:,:] = frame.values[iy, ix]
    return frame

def _shift_matrix_to_index_matrix(shift_matrix):
    """Builds and returns two numpy arrays which are to be used for reindexing.
    
    Parameters
    ----------
    shift_matrix : xarray DataArray
        Shift distance array as returned by construct_shift_matrix().

    Returns
    -------
    index_x : numpy matrix
        New indexes for x direction.
    index_y : numpy matrix
        New indexes for y direction.
    """

    index_x = np.zeros_like(shift_matrix.values,dtype=int)
    index_y = np.zeros_like(shift_matrix.values,dtype=int)

    for x in range(shift_matrix.x.size):
        for y in range(shift_matrix.y.size):
            index_x[y,x] = int(round(x + shift_matrix.values[y,x]))
            index_y[y,x] = y

    # Clamp so that indices won't go out of bounds.
    index_x = np.clip(index_x, 0, x-1)
    return index_x,index_y

def _intr_shift_frame(frame, shift_matrix):
    """ Desmile frame using row-wise interpolation of pixel intensities. """

    ds = xr.Dataset(
        data_vars={
            'frame'      :   frame,
            'x_shift'    :   shift_matrix,
            },
    )

    ds['desmiled_x'] = ds.x - ds.x_shift
    min_x = frame.x.min().item()
    max_x = frame.x.max().item()
    ds.coords['new_x'] = np.linspace(min_x, max_x, frame.x.size)
    ds = ds.groupby('y').apply(_desmile_row)

    ds = ds.drop('x_shift')
    ds = ds.drop('x')
    renames = {'new_x':'x'}
    ds = ds.rename(renames)
    
    return ds.frame

def _intr_shift_cube(cube, shift_matrix):
    """ Desmile cube using row-wise interpolation of pixel intensities.  """

    ds = xr.Dataset(
        data_vars={
            'reflectance'      :   cube.reflectance,
            'x_shift'    :   shift_matrix,
            },
    )
    ds['desmiled_x'] =  ds[P.dim_x] - ds.x_shift

    min_x = cube.reflectance.x.min().item()
    max_x = cube.reflectance.x.max().item()
    ds.coords['new_x'] = np.linspace(min_x, max_x, cube.reflectance.x.size)

    gouped = ds.groupby(P.dim_y)
    ds = gouped.apply(_desmile_row).astype(np.float32)
    ds = ds.drop('x_shift')
    ds = ds.drop(P.dim_x)
    renames = {'new_x':'x'}
    ds = ds.rename(renames)
    # Transpose back into original shape.
    # I was unable to find out why apply() switches the 
    # dimensions to (y, index, x)
    ds = ds.transpose(P.dim_scan, P.dim_y, P.dim_x)
    isNan = np.isnan(ds.reflectance.values).any()
    if isNan:
        print(f"Interpolatively shifted cube contains NaNs.")
    isInf = np.isinf(ds.reflectance.values).any()
    if isInf:
        print(f"Interpolatively shifted cube contains Infs.")
    # Fix NaNs before comparing for negatives.
    if isNan or isInf:
        ds.reflectance.values = np.nan_to_num(ds.reflectance.values).astype(np.float32)
    isNeg = (ds.reflectance.values < 0.0).any()
    if isNeg:
        print(f"Interpolatively shifted cube contains negative values.")
        ds.reflectance.values = np.clip(ds.reflectance.values, a_min=0.0).astype(np.float32)
    if isNan or isInf or isNeg:
        isNan = np.isnan(ds.reflectance.values).any()
        print(f"After fixing: Interpolatively shifted cube contains NaNs ({isNan}).")
        isInf = np.isinf(ds.reflectance.values).any()
        print(f"After fixing: Interpolatively shifted cube contains Infs ({isInf}).")
        isNeg = np.any(ds.reflectance.values < 0.0)    
        print(f"After fixing: Interpolatively shifted cube contains negative values ({isNeg}).")
    return ds

def _desmile_row(row):
    """ Interpolate a single row. """

    row['x'] = row.desmiled_x
    new_x = row.new_x
    row = row.drop(['desmiled_x','new_x'])
    row = row.interp(x=new_x, method='linear')
    return row