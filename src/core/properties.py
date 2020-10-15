"""A bunch of constants

"""

path_project_root = '../../'
path_rel_scan = path_project_root + 'scans/'

path_rel_default_cam_settings = path_project_root + 'camera_settings.toml'

extension_camera_settings = '.toml'
extension_control = '.toml'
extension_data_format = '.nc'

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
# Not iplemented everywhere

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



########### Example related #############

path_example_frames =  path_project_root + 'examples/'
example_scan_name = 'example_scan'

ctrl_scan_settings = 'scan_settings'
ctrl_scanning_speed = 'scanning_speed'
ctrl_width = 'width'
ctrl_width_offset = 'width_offset'
ctrl_height = 'height'
ctrl_height_offset = 'height_offset'

ctrl_spectral_lines = 'spectral_lines'
ctrl_positions = 'positions'
ctrl_wavelengths = 'wavelengths'
ctrl_window_width = 'window_width'
ctrl_peak_width = 'peak_width'

example_scan_control_content =\
f"""
[{ctrl_scan_settings}]
	{ctrl_scanning_speed} = 0.2
	{ctrl_width} 			= 2500
	{ctrl_width_offset} 	= 500
	{ctrl_height}			= 760
	{ctrl_height_offset}	= 975

[{ctrl_spectral_lines}]
    # x-coordinates in full sensor coordinates (not cropped ones). 
	{ctrl_positions} 		= [1130, 1260, 1480, 2020]
	{ctrl_wavelengths} 	= [327, 380, 521]
	{ctrl_window_width} 	= 25
	{ctrl_peak_width} 		= 5
"""
