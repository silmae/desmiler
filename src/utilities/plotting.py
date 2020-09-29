import matplotlib

def get_figure_size():
    return (16,10)

def getColorList(length, mapName='hsv', portion=1.0):
    """Returns a list of colors.

    """
    
    cm = matplotlib.cm.get_cmap(mapName)

    colors = []
    stride = portion / length

    for i in range(length):
        colors.append(cm((1 - portion)  + (i * stride)))
    
    return colors
