import matplotlib.pyplot as plt
import numpy as np

def plot_frame_spectra(original_frame, original_bandpass=None, desmiled_frame=None, desmiled_bandpass=None, window_name=''):
    """Plot spectrum of a frame from top, middle, and bottom.

    If desmiled frame is given, its spectrum will be plotted to the same 
    figure. If bandpass filter is given, it will be plotted too.

    Parameters
    ----------

    """

    if desmiled_frame is not None:
        if original_frame.x.size != desmiled_frame.x.size:
            raise ValueError("Original frame and desmiled frame width is not the same.")
        if original_frame.y.size != desmiled_frame.y.size:
            raise ValueError("Original frame and desmiled frame height is not the same.")
    if original_bandpass is not None:
        if original_frame.x.size != len(original_bandpass[0]):
            raise ValueError("Original frame's width and peak heights list's length is not the same.")
    if desmiled_bandpass is not None:
        if original_frame.x.size != len(desmiled_bandpass[0]):
            raise ValueError("Original frame's width and peak heights list's length is not the same.")

    w = original_frame.x.size
    h = original_frame.y.size
    lw = 1

    xData = np.arange(w)
    
    # always do two columns even if no desmile frame, because subplots() is a bitch
    num = 'bandpass ' + window_name
    _,ax = plt.subplots(num=num,ncols=2)
    
    ax[0].set_title("Original")
    ax[0].plot(xData, original_frame.isel(y=int(2*h/3)).values,linewidth=lw,color='c')
    ax[0].plot(xData, original_frame.isel(y=int(h/2)).values,linewidth=lw,color='g')
    ax[0].plot(xData, original_frame.isel(y=int(h/3)).values,linewidth=lw,color='y')
    if original_bandpass is not None:
        ax[0].plot(xData, original_bandpass[0],linewidth=lw,color='b')
        ax[0].plot(xData, original_bandpass[1],linewidth=lw,color='r')

    if desmiled_frame is not None:
        ax[1].set_title("Desmiled")            
        ax[1].plot(xData, desmiled_frame.isel(y=int(2*h/3)).values,linewidth=lw,color='c')
        ax[1].plot(xData, desmiled_frame.isel(y=int(h/2)).values,linewidth=lw,color='g')
        ax[1].plot(xData, desmiled_frame.isel(y=int(h/3)).values,linewidth=lw,color='y')
        if desmiled_bandpass is not None:
            ax[1].plot(xData, desmiled_bandpass[0],linewidth=lw,color='b')
            ax[1].plot(xData, desmiled_bandpass[1],linewidth=lw,color='r')
    plt.show()