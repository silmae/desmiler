"""

This file contains some frame inspection methods. No class to be instanced as these do not need a state.

"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utilities import plotting
import core.properties as P

def plot_frame_spectra(original_frame, original_bandpass=None, desmiled_frame=None, desmiled_bandpass=None, window_name=''):
    """Plot spectrum of a frame from top, middle, and bottom.

    If desmiled frame is given, its spectrum will be plotted to the same 
    figure. If bandpass filter is given, it will be plotted too.

    This method can be used to visualize generated bandpass filter or just to 
    inspect the spectra of given frame.

    Parameters
    ----------
        original_frame : xarray Dataset
            Frame whose spectra will be plotted.
        original_bandpass : numpy array
            Bandpass filter of given frame. Optional. 
        desmiled_frame : xarray Dataset
            Desmiled frame if one wants to compare it with the original frame. Optional.
        desmiled_bandpass : numpy array
            Bandpass filter of the desmiled frame. Optional.         
        window_name
            Matplotlib window name. Can be used to close the window.
    """

    if desmiled_frame is not None:
        if original_frame[P.dim_x].size != desmiled_frame[P.dim_x].size:
            raise ValueError("Original frame and desmiled frame width is not the same.")
        if original_frame[P.dim_y].size != desmiled_frame[P.dim_y].size:
            raise ValueError("Original frame and desmiled frame height is not the same.")
    if original_bandpass is not None:
        if original_frame[P.dim_x].size != len(original_bandpass[0]):
            raise ValueError("Original frame's width and bandpass list's length is not the same.")
    if desmiled_bandpass is not None:
        if original_frame[P.dim_x].size != len(desmiled_bandpass[0]):
            raise ValueError("Original frame's width and bandpass list's length is not the same.")


    if original_frame[P.naming_frame_data] is not None:
        original_source_frame = original_frame[P.naming_frame_data]
    else:
        original_source_frame = original_frame

    w = original_frame[P.dim_x].size
    h = original_frame[P.dim_y].size
    lw = 1

    xData = np.arange(w)
    
    # always do two columns even if no desmile frame, because subplots() is a bitch
    num = 'bandpass ' + window_name
    _,ax = plt.subplots(num=num,ncols=2, figsize=plotting.get_figure_size())
    
    ax[0].set_title("Original")
    ax[0].plot(xData, original_source_frame.isel(y=int(2*h/3)).values,linewidth=lw,color='c')
    ax[0].plot(xData, original_source_frame.isel(y=int(h/2)).values,linewidth=lw,color='g')
    ax[0].plot(xData, original_source_frame.isel(y=int(h/3)).values,linewidth=lw,color='y')
    if original_bandpass is not None:
        ax[0].plot(xData, original_bandpass[0],linewidth=lw,color='b')
        ax[0].plot(xData, original_bandpass[1],linewidth=lw,color='r')

    if desmiled_frame is not None:

        if original_frame[P.naming_frame_data] is not None:
            desmiled_source_frame = desmiled_frame[P.naming_frame_data]
        else:
            desmiled_source_frame = desmiled_frame

        ax[1].set_title("Desmiled")            
        ax[1].plot(xData, desmiled_source_frame.isel({P.dim_x:int(2*h/3)}).values,linewidth=lw,color='c')
        ax[1].plot(xData, desmiled_source_frame.isel({P.dim_x:int(h/2)}).values,linewidth=lw,color='g')
        ax[1].plot(xData, desmiled_source_frame.isel({P.dim_x:int(h/3)}).values,linewidth=lw,color='y')
        if desmiled_bandpass is not None:
            ax[1].plot(xData, desmiled_bandpass[0],linewidth=lw,color='b')
            ax[1].plot(xData, desmiled_bandpass[1],linewidth=lw,color='r')
    plt.show()

def plot_frame(source, spectral_lines=None, plot_fit_points=False, plot_circ_fit=False,
               plot_line_fit=False, window_name='Frame plot', control=None):
    """Plots the given frame with matplotlib.

    If spectral lines are given, they can be plotted on top of the frame with a 
    few different styles. Name given here can be used to close the plot window.

    Prints frame's metadata to the console if any are found.

    Parameters
    ----------
        source : DataArray or Dataset
            Frame to be plotted. Can be a full dataset containing the frame and optional
            metadata, or just the frame as a DataArray.
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
        control: dict
            Control file so that cropping can be taken into account.
    """

    frame_ds = None
    if source[P.naming_frame_data] is not None:
        frame = source[P.naming_frame_data]
        frame_ds = source
    else:
        frame = source

    height = source[P.dim_y].size

    _,ax = plt.subplots(num=window_name,nrows=2, figsize=plotting.get_figure_size())
    ax[0].imshow(frame, origin='lower')

    if spectral_lines is not None:
        # Colormap
        cmap = cm.get_cmap('PiYG')

        x_offset = 0
        y_offset = 0

        if control is not None:
            x_offset = control[P.ctrl_scan_settings][P.ctrl_width_offset]
            y_offset = control[P.ctrl_scan_settings][P.ctrl_height_offset]

        for i,sl in enumerate(spectral_lines):
            # Change color for every circle
            color = cmap(1 / (i+1) )
            if plot_circ_fit:
                ax[0].add_artist(plt.Circle((sl.circ_cntr_x + x_offset, sl.circ_cntr_y + y_offset),
                                            sl.circ_r, color=color, fill=False))
            if plot_fit_points:
                xx = sl.x + x_offset
                yy = sl.y + y_offset
                ax[0].plot(xx,yy,'.',linewidth=1,color=color)
            if plot_line_fit:
                liny = (sl.line_a*sl.x+sl.line_b) + y_offset
                liny = np.clip(liny, 0, frame[P.dim_y].size)
                ax[0].plot(sl.x, liny, linewidth=1,color=color)

    if frame_ds is not None and len(frame_ds.attrs) >= 1:
        print(f"Frame metadata from Dataset:")
        for key,val in frame_ds.attrs.items():
            print(f"\t{key} : \t{val}")

    if len(frame.attrs) >= 1:
        print(f"Frame metadata from DataArray:")
        for key,val in frame.attrs.items():
            print(f"\t{key} : \t{val}")

    ### Spectrogram
    row_selection = np.linspace(height * 0.1, height * 0.9, num=3, dtype=np.int)
    rows = frame.isel({P.dim_y:row_selection}).values
    rows = rows.transpose()
    ax[1].plot(rows)

    plt.show()
