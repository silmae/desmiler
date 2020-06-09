import numpy as np

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