import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import logging

from utilities import plotting
from core.camera_interface import CameraInterface

class Preview:
    """Class for getting raw live feed from the camera.

    The preview is most useful for focusing the imager to a
    new distance or just to check that all is OK. For focusing,
    use the lower right corner's Variance view while imaging
    some high-contrast target like a black-white checker illuminated
    with white light, such as sun or a halogen spot. The
    focus is best when the variance is as high as possible.
    """

    # Positions of horizontal lines in pixels along y-axis 
    _horizontal_line_positions = None
    # Positions of vertical lines in pixels along x-axis 
    _vertical_line_positions = None
    # Contains all drawable thingys. Even those that are not 
    # updated by snap()
    _plots = []
    # Subplots
    _subplot_cam = None
    _subplot_row_values = None
    _subplot_column_values = None
    _subplot_variance = None

    # How many frames worth of variances are shown.
    _var_frame_count = 1000
    # List of maximum variance history values.
    _mvh_list = None
    # X position of maximum variance in pixels.
    _var_max_pos = 0

    # Animation handle for pyplot function animation. 
    # Used for pausing and unpausing the animation.
    _animation = None

    # Used for animation update
    _fig = None

    # -- State variables --
    # Are plots initialized
    _plots_initialized = False
    # Is live feed (pyplot function animation) running
    _animation_is_running = False

    _cami = None

    def __init__(self, cami:CameraInterface):
        """Initialize the LiveFeed object with given camera and dark frame.
        
        Parameters
        ----------
        cami : CameraInterface object
            Camera to be set for this feed.

        Raises
        ------
        TypeError
            If cam is None
        """

        print(f"Initializing Preview object.")

        self._window_name = 'Preview'
        if cami is None:
            raise TypeError("Camera interface must be provided.")

        self._cami = cami
        self._center_vertical_lines()
        self._center_horizontal_lines()

    def reset(self):
        self.stop()
        try:
            plt.close(self._window_name)
        except:
            logging.error(f"Could not close preview window '{self._window_name}'.")
        self._plots_initialized = False
        self._center_vertical_lines()
        self._center_horizontal_lines()
                    
    def close(self):
        del self._cami
        plt.close(self._window_name)

    def _initPlots(self):
        """Initializes the plots."""

        print("Initializing plots...", end='')

        if self._fig is not None:
            plt.close(self._fig)

        self._plots = []
        self._fig, axs = plt.subplots(nrows=2, ncols=2, num=self._window_name, figsize=plotting.get_figure_size())

        # Positions of the subplots
        self._subplot_cam = axs[0, 0]
        self._subplot_row_values = axs[1, 0]
        self._subplot_column_values = axs[0, 1]
        self._subplot_variance = axs[1, 1]

        self._subplot_cam.set_title('Camera feed')
        self._subplot_cam.set_xlabel('Band')
        self._subplot_cam.set_ylabel('Pixel')
        self._subplot_row_values.set_title('Row intensities')
        self._subplot_row_values.set_xlabel('Band')
        self._subplot_row_values.set_ylabel('Intensity')
        self._subplot_column_values.set_title('Column intensities')
        self._subplot_column_values.set_xlabel('Band')
        self._subplot_column_values.set_ylabel('Intensity')
        self._subplot_variance.set_title('Column variance (focusing)')
        self._subplot_variance.set_xlabel('History')
        self._subplot_variance.set_ylabel('Variance')

        self._cami.turn_on()
        frame = self._cami.get_frame()
        self._cami.turn_off()
        
        self._plots.append(self._subplot_cam.imshow(frame))
        self._fig.colorbar(self._plots[0], ax=self._subplot_cam)

        # Make as many variance graphs as there are vertical lines
        self._mvh_list = []
        for _, _ in enumerate(self._vertical_line_positions):
            self._mvh_list += [np.zeros(self._var_frame_count)]

        line_colors = plotting.getColorList(3,'Oranges',0.7) + plotting.getColorList(3,'Purples',0.7)

        frame_var = frame.var(dim='y')
        for i, _ in enumerate(self._vertical_line_positions):
            self._mvh_list[i][len(self._mvh_list[0]) - 1] = frame_var[self._vertical_line_positions[i]]
            self._plots += self._subplot_variance.plot(
                np.arange(0, len(self._mvh_list[0])),
                self._mvh_list[i],
                color=line_colors[i + len(self._horizontal_line_positions)]
            )        

        frame_x = frame.x.values
        frame_y = frame.y.values
        # Plot of horizontal line values 
        for i, _ in enumerate(self._horizontal_line_positions):
            self._plots += self._subplot_row_values.plot(
                frame_x,
                frame.isel(y=self._horizontal_line_positions[i]).values,
                color=line_colors[i]
                )
        # Plot of vertical line values
        for i, _ in enumerate(self._vertical_line_positions):
            self._plots += self._subplot_column_values.plot(
                frame_y,
                frame.isel(x=self._vertical_line_positions[i]).values,
                color=line_colors[i + len(self._horizontal_line_positions)] # colors are listed so that hor line colors are first
                )

        # These don't have to be updated in snap()
        # Horizontal lines over live feed
        for i, _ in enumerate(self._horizontal_line_positions):
            self._plots += self._subplot_cam.plot(
                frame_x,
                np.ones_like(frame_x) * self._horizontal_line_positions[i],
                '-', 
                linewidth=1, 
                color=line_colors[i]
                )

        # Vertical lines over live feed
        for i, _ in enumerate(self._vertical_line_positions):
            self._plots += self._subplot_cam.plot(
                np.ones_like(frame_y) * self._vertical_line_positions[i],
                frame_y,
                '-', 
                linewidth=1, 
                color=line_colors[i + len(self._horizontal_line_positions)]
                )

        self._plots_initialized = True
        print("... done")
    
    def start(self):
        """Start live feed from the camera. 

        Creates the animation handle.
        Initializes the pots if they are not initialized already.
        """

        if not self._plots_initialized:
            self._initPlots()

        if self._animation is None:
            # blit off so that autoscale can work
            self._animation = animation.FuncAnimation(self._fig, self._snap, interval=10, blit=False)
        else:
            if self._animation.event_source:
                self._animation.event_source.start()
            else:
                logging.warning("No animation event source to start. Creating new animation.")
                self._cami.turn_on()
                self._animation = animation.FuncAnimation(self._fig, self._snap, interval=10, blit=False)
        
        self._animation_is_running = True
        plt.show()

    def stop(self):
        """Stops the live feed from the camera. 

        Stops the animation if it was running and stops camera 
        aquisition.
        """

        if self._animation is not None and self._animation.event_source is not None:
            self._animation.event_source.stop()
            self._animation_is_running = False
            self._cami.turn_off()
    
    def _snap(self, i):
        """Update function for the animation."""

        frame = self._cami.get_frame()

        self._plots[0].set_data(frame.values)

        frame_var = frame.var(dim='y')
        
        for i, _ in enumerate(self._vertical_line_positions):
            # Shift all values one step to the left and place new max variance last.
            self._mvh_list[i] = np.roll(self._mvh_list[i], -1)
            self._mvh_list[i][len(self._mvh_list[0]) - 1] = frame_var[self._vertical_line_positions[i]]
            self._plots[1 + i].set_data(np.arange(0, len(self._mvh_list[0])), self._mvh_list[i])

        self._var_max_pos = frame_var.argmax(dim='x').item()
        frame_max_var = frame_var[self._var_max_pos].item()
        frame_max_px = frame.max().item()
        self._rescale_plots(frame_max_var, frame_max_px)

        # Horizontal lines
        for i, _ in enumerate(self._horizontal_line_positions):
            self._plots[i + 1 + len(self._mvh_list)]\
                .set_data(frame.x.values, frame.isel(y=self._horizontal_line_positions[i]).values)

        # Vertical lines
        for i, _ in enumerate(self._vertical_line_positions):
            self._plots[i + 1 + len(self._mvh_list) + len(self._horizontal_line_positions)]\
                .set_data(frame.y.values, frame.isel(x=self._vertical_line_positions[i]).values)
        
        return self._plots

    def _rescale_plots(self, frame_max_var, frame_max_px):
        """Rescale plots to fit the data. 

        If new maximum exceeds the old ylim for any plot, new ylim is set with some extra 
        space to avoid scaling too often. If new maximum is significanly smaller 
        than the old ylim, new ylim is set to maximum plus some extra space.

        Parameters
        ----------

        frame_max_var:
            is the maximum column-wise variance in current frame
        frame_max_px:
            is the maximum pixel value in current frame. If dark frame 
            is not shot, scaling might not work as expected due to dead pixels in 
            the sensor. 

        """

        extra_space = 1.1
        low_limit = 0.7
        if frame_max_var > self._subplot_variance.get_ylim()[1]:  # or \
            self._subplot_variance.relim()
            self._subplot_variance.set_ylim(0, extra_space * frame_max_var)
            # print(1, "  max: ", frameMaxVar, " new lim: ", self._subPlotVariance.get_ylim()[1])
        if frame_max_var < low_limit * self._subplot_variance.get_ylim()[1]:  # or \
            self._subplot_variance.relim()
            self._subplot_variance.set_ylim(0, extra_space * frame_max_var)
            # print(4, "  max: ", frameMaxVar, " new lim: ", self._subPlotVariance.get_ylim()[1])

        if frame_max_px > self._subplot_column_values.get_ylim()[1]:  # or \
            self._subplot_column_values.relim()
            self._subplot_column_values.set_ylim(0, extra_space * frame_max_px)
            # print(2, "  max: ", frameMaxPx, " new lim: ", self._subPlotColumnValues.get_ylim()[1])
        if frame_max_px < low_limit * self._subplot_column_values.get_ylim()[1]:  # or \
            self._subplot_column_values.relim()
            self._subplot_column_values.set_ylim(0, extra_space * frame_max_px)
            # print(5, "  max: ", frameMaxPx, " new lim: ", self._subPlotColumnValues.get_ylim()[1])
        if frame_max_px > self._subplot_row_values.get_ylim()[1]:  # or \
            self._subplot_row_values.relim()
            self._subplot_row_values.set_ylim(0, extra_space * frame_max_px)
            # print(3, "  max: ", frameMaxPx, " new lim: ", self._subPlotRowValues.get_ylim()[1])
        if frame_max_px < low_limit * self._subplot_row_values.get_ylim()[1]:
            self._subplot_row_values.relim()
            self._subplot_row_values.set_ylim(0, extra_space * frame_max_px)
            # print(6, "  max: ", frameMaxPx, " new lim: ", self._subPlotRowValues.get_ylim()[1])

    def _make_line_positions(self, out, max_dim, center=None,  spacing=None):
        """Centers three lines around a center line."""

        if center is None:
            center = int(max_dim/2)
        if spacing is None:
            spacing = int(max_dim/3)

        out = np.zeros(3,dtype=int)
        out[0] = (center - spacing)
        out[1] = center
        out[2] = (center + spacing)
        np.clip(out, 0, max_dim-1, out=out)

    def _center_horizontal_lines(self, center=None, spacing=None):
        """Centers three horizontal lines around a center line.

        Parameters
        ----------
        center : int
            Row index of the center line.
        spacing : int
            Space (in pixels) to be left between outer line and center line.
        """

        max_dim = self._cami.height()
        self._horizontal_line_positions = np.zeros(3, dtype=int)
        self._make_line_positions(self._horizontal_line_positions, max_dim, center, spacing)

    def _center_vertical_lines(self, center=None, spacing=None):
        """Centers three vertical lines around a center line.

        Parameters
        ----------
        center : int
            Column index of the center line.
        spacing : int
            Space (in pixels) to be left between outer line and center line.
        """

        max_dim = self._cami.width()
        self._vertical_line_positions = np.zeros(3, dtype=int)
        self._make_line_positions(self._vertical_line_positions, max_dim, center, spacing)
