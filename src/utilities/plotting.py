"""

Fairly useless plotting helper. This could made into a complete wrapper around matplotlib to do stuff properly,
but ofcource we have no time for such undertaking.

"""

import matplotlib

def get_figure_size():
    """Hard-coded figure size. Pretty stupid, really.

    TODO move this to parameter file.
    """
    return (16,10)

def getColorList(length, map_name='hsv', portion=1.0):
    """Returns a list of colors form given colormap.

    Parameters
    ----------
        length
            How many colors the resulting color list should contain.
        map_name: string, optional, default 'hsv'
            Name of the matplotlib colormap to be used.
        portion: float, optional, default = 1.0
            If less than one, the portion length is cut out from both
            ends of the colormap. This is because some maps have almost white
            ends, which do not show well in white background.

    Returns
    -------
        A list of colors of length length.
    """
    
    cm = matplotlib.cm.get_cmap(map_name)

    colors = []
    stride = portion / length

    for i in range(length):
        colors.append(cm((1 - portion) + (i * stride)))
    
    return colors
