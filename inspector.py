import matplotlib.cm as cm
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

def plot_frame(frame, spectral_lines=None, plot_fit_points=False, plot_circ_fit=False, 
                plot_line_fit=False, window_name='Frame plot'):
    """Plots the given frame with matplotlib.

    If spectral lines are given, they can be plotted on top of the frame with a 
    few different styles. Name given here can be used to close the plot window.

    Parameters
    ----------
    frame : DataArray
        Frame to be plotted.
   
    spectral_lines : list of SpectralLine objects
        If given, frame is overlayed with spectral lines.
    plot_fit_points:
        If True, frame is overlayed with points used to fit the spectral line.
    plot_circ_fit:
        If True, frame is overlayed with fitted circle arcs.
    plot_line_fit:
        If True, frame is overlayed with fitted lines. 
    window_name : string,int
        Name for the plotting window. Use plt.close(<window_name>) to close the window 
        if needed.    
    """

    fig,ax = plt.subplots(num=window_name)
    ax.imshow(frame, origin='lower')
    ax.set_ylim(0,frame.y.size)

    if spectral_lines is not None:
        # Colormap
        cm = cm.get_cmap('PiYG')
    
        for i,line in enumerate(spectral_lines):
            # Change color for every circle
            color = cm(1 / (i+1) )
            if plot_circ_fit:
                ax.add_artist(plt.Circle((line.circ_cntr_x, line.circ_cntr_y), line.circ_r, color=color, fill=False))
            if plot_fit_points:
                ax.plot(line.x,line.y,'.',linewidth=1,color=color)
            if plot_line_fit:
                liny = line.a*line.x+line.b
                ax.plot(line.x, liny, linewidth=1,color=color)

    plt.show()