"""

A bunch of constants used throughout the program.

"""

# Default paths
path_project_root = '../'
path_rel_scan = path_project_root + 'scans/'
path_rel_default_cam_settings = path_project_root + 'camera_settings.toml'
path_example_frames =  path_project_root + 'examples/'

# Used file extensions
extension_camera_settings = '.toml'
extension_control = '.toml'
extension_data_format = '.nc'

# Expected filenames
fn_camera_settings = 'camera_settings' + extension_camera_settings
fn_control = 'control' + extension_control
ref_dark_name = 'dark'
ref_white_name = 'white'
ref_light_name = 'light'
shift_name = 'shift'
cube_raw_name = 'raw'
cube_reflectance_name = 'rfl'
cube_desmiled_lut = 'desmiled_lut'
cube_desmiled_intr = 'desmiled_intr'
example_scan_name = 'example_scan'

freeform_session_name = 'freeform'

# Default count of frames for dark, white, and peak light frames.
dwl_default_count = 10
# Default reduction method (mean or median) for dark, white, and peak light frames.
dwl_default_method = 'mean'

# A single frame dataset uses this name to save the frame.
naming_frame_data = 'frame'
naming_cube_data = 'dn'
naming_dark_corrected = 'dark_corrected'
naming_reflectance = 'reflectance'

########### Dimension names #############

dim_x = 'x'
dim_y = 'y'
dim_scan = 'scan_index'

dim_order_frame = dim_y,dim_x
dim_order_cube = dim_scan,dim_y,dim_x

########### Metadata keys   #############

meta_key_sl_count = 'sl_count'
meta_key_location = 'location'
meta_key_tilt = 'tilt'
meta_key_curvature = 'curvatures'
meta_key_sl_X = 'sl_X'
meta_key_sl_Y = 'sl_Y'

########### Control file keys #############

ctrl_cube_inspector = 'cube_inspector'
ctrl_spectral_filter = 'spectral_filter'
ctrl_spectral_blue = 'spectral_blue'
ctrl_spectral_green = 'spectral_green'
ctrl_spectral_red = 'spectral_red'

ctrl_scan_settings = 'scan_settings'
ctrl_is_mock_scan = 'is_mock_scan'
ctrl_scanning_speed_value   = 'scanning_speed_value'
ctrl_scanning_speed_unit    = 'scanning_speed_unit'
ctrl_scanning_length_value  = 'scanning_length_value'
ctrl_exporure_time_s  = 'exposure_time_s'
ctrl_acquisition_overhead = 'acquisition_overhead'

ctrl_width = 'width'
ctrl_width_offset = 'width_offset'
ctrl_height = 'height'
ctrl_height_offset = 'height_offset'

ctrl_spectral_lines = 'spectral_lines'
ctrl_positions = 'positions'
ctrl_wavelengths = 'wavelengths'
ctrl_window_width = 'window_width'
ctrl_peak_width = 'peak_width'

########### Default control file content #############

example_scan_control_content =\
f"""

# Settings related to the CubeInspector. 
[{ctrl_cube_inspector}]
    # Spectral filter to be used with SAM
    {ctrl_spectral_filter}      = [500,1000]

    # Define spectral areas of RGB to be used in false color image construction.
    # Use cube inspector's spectrogram to find correct areas.
    {ctrl_spectral_blue}        = [300, 500]
    {ctrl_spectral_green}       = [660, 860]
    {ctrl_spectral_red}         = [1300, 1500]

# Settings related to the scan. 
[{ctrl_scan_settings}]

    # Use mock scan to check that your values are correct. Values 0 = False, 1 = True. 
    {ctrl_is_mock_scan}           = 0

    # Speed in arbitrary length unit per second.
    {ctrl_scanning_speed_value  } = 10.0

    # Length in same units as speed. 
    {ctrl_scanning_length_value } = 20.0

    # Exposure time in seconds. (Note that exposure in camera settings are in microseconds.)
    {ctrl_exporure_time_s}        = 0.02

    # There is some overhead when acquiring a frame and processing it into the image cube. 
    # The over head defined here as a percentage will be added to exposure time in frame acquisition 
    # loop to ensure that there is enough time to capture all the frames. The program will tell you if 
    # the overhead is too big or too low. You can verify it only by running an actual scan, i.e. 
    # mock = False.  
    {ctrl_acquisition_overhead}   = 0.10

    # Rest of the settings are for cropping the acquired frames. Note that the camera can provide 
    # frames more rapidly if it does not have to pass on full sensor sized frames (less data is 
    # transferred), which means that you can run scans at higher speeds.
    # NOTE! Use values dividable by 2. There will be unhandled errors otherwise.
	{ctrl_width} 			= 2500
	{ctrl_width_offset} 	= 500
	{ctrl_height}			= 760
	{ctrl_height_offset}	= 976

# Settings related to spectral lines and desmiling
[{ctrl_spectral_lines}]
    # Spectral line positions as x-coordinates of the frame. Use preview functionality to manually find 
    # some (or one) of the most bright spectral lines. 
    # NOTE! Spectral line positions are expected to be in non-cropped, i.e., full sensor sized frame 
    # coordinates. 
	{ctrl_positions} 		= [1130, 1260, 1480, 2020]

    # TODO Wavelength calibration not yet implemented.
	#{ctrl_wavelengths} 	= [327, 380, 521]

    ### Peak finding ###

    # Half of the search window width in pixels. Searh window is centered around given spectral line 
    #positions for the peak finding algorithm.  
	{ctrl_window_width} 	= 25

    # Required peak width in pixels. 
	{ctrl_peak_width} 		= 5
"""
