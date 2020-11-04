import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

import core.properties as P

from core import scan as scan

# Under what name the data is in the cube. 
cube_data_name = 'dn'

# General calculations to form false color images, spectral angle maps, etc.
# --------------------------

def calculate_false_color_images(org, lut, intr, viewable):
    """ Calculate false color images for all three cubes.

    Expects to find color checker blue, green, and red tile from 
    hard coded areas.
    """

    rBand = np.arange(1300,1500)
    gBand = np.arange(660,860)
    bBand = np.arange(300, 500)
    rgb = np.array([rBand, gBand, bBand])
    # Assumes that dimensions are ordered (P.dim_scan, P.dim_y, P.dim_x)
    org_mean = np.mean(org[viewable].values[:,:,rgb], axis=3).astype(np.float32)
    lut_mean = np.mean(lut[viewable].values[:,:,rgb], axis=3).astype(np.float32)
    intr_mean = np.mean(intr[viewable].values[:,:,rgb], axis=3).astype(np.float32)
    org_false = (org_mean / np.max(org_mean, axis=(0,1))).clip(min=0.0)
    lut_false = (lut_mean / np.max(lut_mean, axis=(0,1))).clip(min=0.0)
    intr_false = (intr_mean / np.max(intr_mean, axis=(0,1))).clip(min=0.0)
    return org_false, lut_false, intr_false

def calculate_sam(sourceCube, sam_window_start, sam_window_end, sam_ref_x, viewable,
                use_radians=False, spectral_filter=None, use_scm=False):
    """Calculates spectral angle map of a single cube.
    
    Parameters
    ----------
        sourceCube : DataSet
            Spectral cube from which to calculate the cosine angle.
        sam_window_start : list of ints of lengts 2 (x,y)
            Starting coordinate of the window from where the cosine angle is calculated.
        sam_window_end : list of ints of lengts 2 (x,y)
            Ending coordinate of the window from where the cosine angle is calculated.
        sam_ref_x : int
            x-coordinate of the reference point inside the window.
        use_radians : bool, optional default=False
            If True, the actual cosine angle is calculated, else only raw dot product 
            is calculated. Raw dot product is numerically more accurate.
         spectral_filter : slice
            If given, bands from slice.start to slice.stop are used for cosine angle 
            calculations. Otherwise, the whole spectrum is used.
        use_scm : bool, optional default False
            SCM is advanced version of the basic SAM spectral angle.
    Returns 
    -------
        sam : numpy array
            Plottable spectral angle map image of the size of y,index dimensions of the source cube.
            Values outside of given window are filled with ones, if use_radians = True, and 
            zeros otherwise.
        chunk : DataArray
            Calculated cosine angels the size of given sam window. This is a subset of sam.
    """

    x_slice = slice(sam_window_start[0], sam_window_end[0])
    y_slice = slice(sam_window_start[1], sam_window_end[1] )

    if spectral_filter is not None:
        sf = spectral_filter
    else:
        sf = slice(0, sourceCube.reflectance.x.size-1)

    cos_ref = np.clip(sam_ref_x, sam_window_start[0], sam_window_end[0]) 

    # Reference spectrum as mean of a vertical line in the box.

    a = sourceCube[viewable].isel({P.dim_y: cos_ref, P.dim_scan: y_slice, P.dim_x:sf}).mean(dim=P.dim_scan).astype(np.float64)
    # Reference spectrum as a single pixel spectrum
    # a = sourceCube.reflectance.isel(y=cos_ref, index=int((sam_window_end[1]-sam_window_start[1])/2), x=sf)
    # Reference point as a mean over the whole box area
    # a = sourceCube.reflectance.isel(y=x_slice, index=y_slice, x=sf).mean(dim=('y','index'))
    b = sourceCube[viewable].isel({P.dim_y: x_slice, P.dim_scan: y_slice, P.dim_x:sf}).astype(np.float64)

    if not use_scm:
        ###    SAM    ####
        chunk = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b, axis=2))
    else:
        ###    SCM    ####
        aMean = a.mean(dim=P.dim_x)
        bMean = b.mean(dim=P.dim_x)
        A = a - aMean
        B = b - bMean
        chunk = A.dot(B) / ((np.linalg.norm(A) * np.linalg.norm(B, axis=2)))
        chunk = chunk / 2
        chunk = chunk + 0.5

    if use_radians:
        # Safeguard for arccos.
        chunk = chunk.clip(min=0.0,max=1.0)
        chunk = np.arccos(chunk)
        chunk = chunk.rename('cosine angle')
        # Initialize cosmap image with zeros
        sam = np.zeros_like(sourceCube[viewable].isel({P.dim_x:0}).values).astype(np.float64)
    else:
        chunk = chunk.rename('dot product')
        # Initialize cosmap image with ones
        sam = np.ones_like(sourceCube[viewable].isel({P.dim_x:0}).values).astype(np.float64)

    sam[y_slice, x_slice] = chunk

    return sam, chunk

class CubeInspector:
    """CubeInspector is an interactive matplotlib-based inspector program with simple key and mouse commands.
    
    """

    def __init__(self, org, lut, intr, viewable, control=None):
        super().__init__()
        self.org = org
        self.lut = lut
        self.intr = intr
        self.viewable = viewable
        # Interpolative shift may cause very small negative values, which should be clipped. 
        # self.intr[self.viewable].values = self.intr[self.viewable].values.clip(min=0.0).astype(np.float32)

        self.width_image = self.org[self.viewable][P.dim_y].size
        self.height_image = self.org[self.viewable][P.dim_scan].size

        # Selected pixel and band in CUBE's coordinates. show() deals with the 
        # transformation from plot coordinates.
        self.idx = int(self.org[self.viewable][P.dim_scan].size / 2) # image y
        self.y = int(self.org[self.viewable][P.dim_y].size / 2) # image x
        self.x = int(self.org[self.viewable][P.dim_x].size / 2) # image band
        # Reference point for spectral angle. In same dimension as self.y.
        self.sam_ref_x = self.y

        # Modes: 1 for reflectance image, 2 for false color image, 3 for spectral angle
        self.mode = 1

        # Containers for false color images
        self.org_false = None
        self.lut_false = None
        self.intr_false = None
        # False color images calculated lazyly only once.
        self.false_color_calculated = False
        # Toggle mode 3 between radians and dot product.
        self.toggle_radians = False

        # Matplotlib figure and axis.
        self.fig = None
        self.ax = None
        # First call to show() will initialize plots.
        self.plot_inited = False

        # Color palettes. Colors in colors_org_lut_intr are used for original, 
        # reindexed, and interpolated desmile in line plots and headers.
        # color_pixel_selection is used for pixel and band selection indicators.
        self.colors_rbg = ['red','green','blue']
        self.colors_org_lut_intr = ['black','lightcoral','purple']
        self.color_pixel_selection = 'violet'
                
        # Boundaries of RGB boxes used for false color calculations.
        lins = np.linspace(0, self.height_image, num=13, dtype=np.int)
        self.rgb_horizontal_chunk = slice(int(self.width_image/10), self.width_image - int(self.width_image/10))
        self.rgb_vertical_chunks = [slice(lins[1], lins[3]), slice(lins[5], lins[7]), slice(lins[9], lins[11])]
        
        # Images to be plotted on update. Active mode will stuff images in 
        # this list, which are then drawn over the old ones.
        self.images = []
        
        # Stores cosine maps (org,rei,dei) of the selected area.
        self.sam_chunks_list = []

        # Filter out noisy ends of the spectrum in cosine maps.
        self.spectral_filter_max = self.org[self.viewable][P.dim_x].size
        # Filter one third from the middle of the spectrum by default.
        lin_spectr = np.linspace(0, self.spectral_filter_max, 4, dtype=np.int)
        self.spectral_filter = slice(lin_spectr[1], lin_spectr[2])
        # How much the spectral filter is moved to left or right.        
        self.spectral_filter_step = 100

        # Cosine boxes
        self.sam_window_start = \
            [int(self.width_image/2)-int(self.width_image/4),int(self.height_image/2)-int(self.height_image/4)]
        self.sam_window_end = \
            [int(self.width_image/2)+int(self.width_image/4),int(self.height_image/2)+int(self.height_image/4)]
        self.sam_window_start_sug = [self.sam_window_start[0], self.sam_window_start[1]]
        self.sam_window_corner_given = False

        # Image overlay decorations. Store handles for removing the objects.
        self.decorations_selection = []
        self.decorations_box = []

    def init_plot(self):
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, num='Cube', figsize=(16,12))
        self.connect_ui()
        self.init_images()
        self.plot_inited = True
    
    def init_images(self):
        print("Initializing images...", end=' ', flush=True)
        self.images.append(self.org[self.viewable].isel({P.dim_x:self.x}).plot.imshow(ax=self.ax[0,1]))
        self.images.append(self.lut[self.viewable].isel({P.dim_x:self.x}).plot.imshow(ax=self.ax[1,1]))
        self.images.append(self.intr[self.viewable].isel({P.dim_x:self.x}).plot.imshow(ax=self.ax[1,0]))
        print("done")

    def connect_ui(self):
        self.connection_mouse = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.connection_button = self.fig.canvas.mpl_connect('key_press_event', self.keypress)

    def disconnect_ui(self):
        self.fig.canvas.mpl_disconnect(self.connection_mouse)
        self.fig.canvas.mpl_disconnect(self.connection_button)

    def onclick(self, event):
        """ Handle mouse clicks.

        Usage
        -----
            alt + click
                Ignores the click. Allows using matplotlib's own commands 
                such as zooming without interference.
            shift + click
                First time: set the starting corner of a cosine box, second click 
                finishes the selection and triggers image update. Any other click 
                between these two will reset the selection.
            click on line plot area in mode 1
                Select new band and trigger image update.
            click on any image in modes 1 or 2
                Select new pixel to plot spectra for.
            click on any image in mode 3
                Select new reference point for cosine angle calculations.
        """

        if event.key == 'alt':
            return
        if event.inaxes is not None:
            # Clicking line plot triggers band change in mode 1
            if event.inaxes is self.ax[0,0] and self.mode == 1:
                self.show(band=int(event.xdata))
            # Clicking one of the three images..
            else:
                # Shift clicks set boxes for cosine calculations.
                if event.key == 'shift':
                    # First shift click selects first corner of cos box
                    if not self.sam_window_corner_given:
                        self.sam_window_start_sug = [int(event.xdata), int(event.ydata)]
                        self.sam_window_corner_given = True
                    # Second click finalizes the box and triggers an update.
                    else:
                        startX = self.sam_window_start_sug[0]
                        startY = self.sam_window_start_sug[1]
                        x = int(event.xdata)
                        y = int(event.ydata)
                        if x < startX:
                            temp = x
                            x = startX
                            startX = temp
                        if y < startY:
                            temp = y
                            y = startY
                            startY = temp
                        self.sam_window_start = [startX, startY]
                        self.sam_window_end = [x, y]
                        # reset status
                        self.sam_window_corner_given = False
                        self.sam_window_start_sug = [startX, startY]
                        # If in cosmap mode, force update in show()
                        if self.mode == 3:
                            # Update cos box without updating the reference point
                            self.show(force_update=True)
                else:
                    # any other update command resets cos box status
                    self.sam_window_corner_given = False
                    self.sam_window_start_sug = [self.sam_window_start[0], self.sam_window_start[1]]
                    if self.mode == 3:
                        self.show(sam_ref_x=int(event.xdata))
                    else:
                        self.show(x=int(event.xdata), y=int(event.ydata))

    def keypress(self, event):
        """ Handles key presses.

        Usage
        -----
            numkey 1
                Select reflectance image mode.
            numkey 2
                Select false color image mode.
            numkey 3
                Select cosine angle mode.
            r
                Toggle between dot product and cosine angle in mode 3.
            a
                If in mode 3, move spectral filter to the left
            d
                If in mode 3, move spectral filter to the right
        """

        key = event.key
        if key == '1' or key == '2' or key == '3':
            self.mode = int(key)
            self.show(mode=self.mode)
        if key == 'r':
            self.toggle_radians = not self.toggle_radians
            self.show(force_update=True)
        if self.mode == 3:      # How much the spectral filter is moved to left or right.            
            if key == 'a':      
                low = self.spectral_filter.start - self.spectral_filter_step
                high = self.spectral_filter.stop - self.spectral_filter_step
                low = np.clip(low, a_min=0, a_max=self.spectral_filter_max)
                high = np.clip(high, a_min=0, a_max=self.spectral_filter_max)
                self.spectral_filter = slice(low, high)
                self.show(force_update=True)        
            if key == 'd':      
                low = self.spectral_filter.start + self.spectral_filter_step
                high = self.spectral_filter.stop + self.spectral_filter_step
                low = np.clip(low, a_min=0, a_max=self.spectral_filter_max)
                high = np.clip(high, a_min=0, a_max=self.spectral_filter_max)
                self.spectral_filter = slice(low, high)
                self.show(force_update=True)
    
    def show(self, x=None, y=None, band=None, mode=None, force_update=False, sam_ref_x=None, spectral_filter=None):
        """Update images, spectrograms, and overlays as needed."""

        band_changed = False
        mode_changed = False
        cos_ref_x_changed = False
        spectral_filter_changed = False

        if x is not None:
            self.y = x
        if y is not None:
            self.idx = y
        if band is not None:
            self.x = band
            band_changed = True
        if sam_ref_x is not None:
            # This is usually right unless cosine box has been resized
            self.sam_ref_x = np.clip(sam_ref_x, self.sam_window_start[0], self.sam_window_end[0]) 
            cos_ref_x_changed = True

        acceptedModes = [1,2,3]
        if mode is not None:
            if mode in acceptedModes:
                self.mode = mode
                mode_changed = True
            elif self.mode in acceptedModes:
                print(f"Mode {mode} not allowed. Staying in mode {self.mode}.")
            else:
                fallback_mode = 1
                print(f"Mode {mode} not allowed. Falling back to mode {fallback_mode}.")                
                self.mode = fallback_mode
                mode_changed = True
        
        if not self.plot_inited:
            self.init_plot()
            force_update = True      

        # No need to update images if only pixel selection or cosine reference is changed.
        if band_changed or mode_changed or cos_ref_x_changed or force_update:
            self.update_images()

        self.update_spectrograms()
        self.update_image_overlay()
        plt.show()
        plt.draw() #redraw
    
    def update_images(self):
        """Updates images depending on selected mode.

        May call expensive calculations, so call only when needed.
        """

        # print(f"Updating images (mode {self.mode})...", end=' ', flush=True)

        if self.mode == 1:
            source = self.org[self.viewable].isel({P.dim_x:self.x})
            self.images[0].set_data(source)
            self.images[0].set_norm(cm.colors.Normalize(source.min(), source.max()))
            source = self.lut[self.viewable].isel({P.dim_x:self.x})
            self.images[1].set_data(source)
            self.images[1].set_norm(cm.colors.Normalize(source.min(), source.max()))
            source = self.intr[self.viewable].isel({P.dim_x:self.x})
            self.images[2].set_data(source)
            self.images[2].set_norm(cm.colors.Normalize(source.min(), source.max()))          
           
            self.ax[0,1].set_title(f'ORG, band={self.x}', color=self.colors_org_lut_intr[0])
            self.ax[1,1].set_title(f'LUT, band={self.x}', color=self.colors_org_lut_intr[1])
            self.ax[1,0].set_title(f'INTR, band={self.x}', color=self.colors_org_lut_intr[2])
        elif self.mode == 2:
            if not self.false_color_calculated:
                self.org_false, self.lut_false, self.intr_false = calculate_false_color_images(self.org, self.lut, self.intr, self.viewable)
                self.false_color_calculated = True

            self.images[0].set_data(self.org_false)
            self.images[1].set_data(self.lut_false)
            self.images[2].set_data(self.intr_false)
            self.ax[0,1].set_title(f'ORG false color picture', color=self.colors_org_lut_intr[0])
            self.ax[1,1].set_title(f'LUT false color picture', color=self.colors_org_lut_intr[1])
            self.ax[1,0].set_title(f'INTR false color picture', color=self.colors_org_lut_intr[2])
        elif self.mode == 3:
            self.calculate_sams()
            if self.toggle_radians:
                cosType = 'cosine angle'
            else:
                cosType = 'normalized dot product'
            self.ax[0,1].set_title(f'ORG {cosType}', color=self.colors_org_lut_intr[0])
            self.ax[1,1].set_title(f'LUT {cosType}', color=self.colors_org_lut_intr[1])
            self.ax[1,0].set_title(f'INTR {cosType}', color=self.colors_org_lut_intr[2])

        # print("done")

    def update_spectrograms(self):
        """Update spectrogram view (top left) and its overlays."""

        # print(f"Updating spectrograms...", end=' ', flush=True)
        
        self.ax[0,0].clear()
        if self.mode == 1 or self.mode == 2:
            self.org[self.viewable].isel({P.dim_y:self.y, P.dim_scan:self.idx}).plot(ax=self.ax[0,0], color=self.colors_org_lut_intr[0])
            self.lut[self.viewable].isel({P.dim_y:self.y, P.dim_scan:self.idx}).plot(ax=self.ax[0,0], color=self.colors_org_lut_intr[1])
            self.intr[self.viewable].isel({P.dim_y:self.y, P.dim_scan:self.idx}).plot(ax=self.ax[0,0], color=self.colors_org_lut_intr[2])

            # Reference color spectra
            for i,_ in enumerate(self.rgb_vertical_chunks):
                rgb_chunk = self.org[self.viewable].isel({P.dim_y:self.rgb_horizontal_chunk, P.dim_scan:self.rgb_vertical_chunks[i]}).mean(dim=(P.dim_scan, P.dim_y))
                rgb_chunk.plot(ax=self.ax[0,0], color=self.colors_rbg[i])

            self.ax[0,0].set_title('Spectrograms')
        else:
            if self.toggle_radians:
                cosType = 'Mean cosine angle'
                yLabel = 'angle [rad]'
            else:
                cosType = 'Mean normalized dot product'
                yLabel = 'dot product'
            self.ax[0,0].set_title(f"{cosType}, bands {self.spectral_filter.start} - {self.spectral_filter.stop}")
            self.ax[0,0].set_ylabel(yLabel)

        if self.mode == 1:
            #Redraw band selection indicator
            _,ylim = self.ax[0,0].get_ylim()
            ylim = int(ylim)
            xd = np.ones(2)*self.x
            yd = np.array([0, np.max(self.org[self.viewable].isel({P.dim_y: self.y, P.dim_scan: self.idx}))])
            self.ax[0,0].plot(xd,yd,color=self.color_pixel_selection)
        elif self.mode == 3:
            # Draw mean of each cos box.
            for i,chunk in enumerate(self.sam_chunks_list):
                xx = np.arange(len(chunk.y))
                yy = np.mean(chunk, axis=0)
                self.ax[0,0].plot(xx, yy, color=self.colors_org_lut_intr[i])
        
        # print("done")

    def update_image_overlay(self):
        """Update decorations drawn over the images."""

        # print(f"Updating image overalys...", end=' ', flush=True)
        
        # Remove decoration boxes from the matplotlib images.
        for i,_ in enumerate(self.decorations_box):
            self.decorations_box[i].remove()

        # And clear them from the storage as well.
        self.decorations_box = []

        if self.mode == 1 or self.mode == 2:
            for i,rgbXChunk in enumerate(self.rgb_vertical_chunks):
                bottomLeftCorner = (self.rgb_horizontal_chunk.start, rgbXChunk.start)
                w = self.rgb_horizontal_chunk.stop - self.rgb_horizontal_chunk.start
                h = rgbXChunk.stop-rgbXChunk.start
                rgbBox = patches.Rectangle(bottomLeftCorner, w, h, edgecolor=self.colors_rbg[i],facecolor='none', linewidth=1)
                self.decorations_box.append(rgbBox)
        elif self.mode == 3:            
            cos_ref = np.clip(self.sam_ref_x, self.sam_window_start[0], self.sam_window_end[0]) 
            halfBox1 = patches.Rectangle((self.sam_window_start[0], self.sam_window_start[1]), cos_ref-self.sam_window_start[0], self.sam_window_end[1]-self.sam_window_start[1], edgecolor='white',facecolor='none', linewidth=1)
            halfBox2 = patches.Rectangle((cos_ref, self.sam_window_start[1]), self.sam_window_end[0]-cos_ref, self.sam_window_end[1]-self.sam_window_start[1], edgecolor='white',facecolor='none', linewidth=1)
            self.decorations_box.append(halfBox1)
            self.decorations_box.append(halfBox2)

        for i,box in enumerate(self.decorations_box):
            self.ax[0,1].add_patch(box)

        for i,scat in enumerate(self.decorations_selection):
            self.decorations_selection[i].remove()
        self.decorations_selection = []

        # Add pixel selection dot for modes 1 and 2
        if self.mode == 1 or self.mode == 2:
            for i in range(2):
                for j in range(2):
                    if i==0 and j==0:
                        continue
                    selection = self.ax[i,j].scatter([self.y], [self.idx], color=self.color_pixel_selection)
                    self.decorations_selection.append(selection)

        # print("done")

    def calculate_sams(self):
        """Calculates and saves spectral angle maps for all three cubes and sets them to image list."""

        sams = []
        self.sam_chunks_list = []
        sam, cosMapChunk = calculate_sam(self.org, self.sam_window_start, self.sam_window_end,
                                         self.sam_ref_x, self.viewable, self.toggle_radians, self.spectral_filter)
        sams.append(sam)
        self.sam_chunks_list.append(cosMapChunk)
        sam, cosMapChunk = calculate_sam(self.lut, self.sam_window_start, self.sam_window_end,
                                         self.sam_ref_x, self.viewable, self.toggle_radians, self.spectral_filter)
        sams.append(sam)
        self.sam_chunks_list.append(cosMapChunk)
        sam, cosMapChunk = calculate_sam(self.intr, self.sam_window_start, self.sam_window_end,
                                         self.sam_ref_x, self.viewable, self.toggle_radians, self.spectral_filter)
        sams.append(sam)
        self.sam_chunks_list.append(cosMapChunk)

        maxVal = np.max(np.array(list(np.max(chunk) for chunk in self.sam_chunks_list)))

        for i,sam in enumerate(sams):
            self.images[i].set_data(sam)
            self.images[i].set_norm(cm.colors.Normalize(sam.min(), maxVal))


if __name__ == '__main__':
    
    ci = CubeInspector(scan_name='test_scan_1')
    ci.show()