"""A bunch of constants

"""

path_project_root = '../'
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

freeform_session_name = 'freeform'

# Default count of frames for dark, white, and peak light frames.
dwl_default_count = 10
# Default reduction method (mean or median) for dark, white, and peak light frames.
dwl_default_method = 'mean'

# A single frame dataset uses this name to save the frame.
naming_frame_data = 'frame'

########### Dimension names #############
# Not iplemented everywhere

dim_x = 'x'
dim_y = 'y'
dim_scan = 'scan_index'

########### Example related #############

example_scan_name = 'example_scan'
example_scan_control_content =\
"""
[scan_settings]
	scanning_speed = 0.2
	width 			= 2500
	width_offset 	= 500
	height			= 760
	height_offset	= 975

[spectral_lines]
    # x-coordinates in full sensor coordinates (not cropped ones). 
	positions 		= [1130, 1260, 1480, 2020]
	wavelengths 	= [327, 380, 521]
	window_width 	= 25
	peak_width 		= 5
"""
