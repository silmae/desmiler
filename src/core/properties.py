"""A bunch of constants

"""

path_rel_scan = '../scans'
frame_folder_name = '../frames'

camera_settings_file_name = 'camera_settings.toml'
control_file_name = 'control.toml'

extension_dark = 'dark'
extension_white = 'white'
extension_light = 'light'

path_rel_default_cam_settings = '../camera_settings.toml'

freeform_session_name = 'freeform'

# Default count of frames for dark, white, and peak light frames.
dwl_default_count = 10
# Default reduction method (mean or median) for dark, white, and peak light frames.
dwl_default_method = 'mean'

# A single frame dataset uses this name to save the frame.
naming_frame_data = 'frame'

example_scan_name = 'example_scan'
example_scan_control_content =\
"""
[scan_settings]
	scanning_speed = 0.2
	width 			= 20001
	width_offset 	= 0
	height			= 1200
	height_offset	= 400

[spectral_lines]
	positions 		= [50, 100, 183]
	wavelengths 	= [327, 380, 521]
	window_width 	= 25
	peak_width 		= 3
"""
