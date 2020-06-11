import xarray as xr
import os

import smile_correction as smile
import inspector as insp

if __name__ == '__main__':

    # Single frames
    # -------------
    test_frame_name = 'test_frame'
    test_frame_sl_locations = [230,267,319,369,439,525,635,681,733,793,840,978,1030,1212,1400]
    original_frame = xr.open_dataset(os.path.normpath('frames/' + test_frame_name + '.nc')).frame
    bp = smile.construct_bandpass_filter(original_frame, test_frame_sl_locations, 30)
    sl_list = smile.construct_spectral_lines(original_frame, test_frame_sl_locations, bp)
    shift_matrix = smile.construct_shift_matrix(sl_list, original_frame.x.size,  original_frame.y.size)

    # Test and plot desmiling with LUT method
    desmiled_frame = smile.apply_shift_matrix(original_frame, shift_matrix, target_is_cube=False, method=0)
    insp.plot_frame(original_frame, sl_list, True, True, True)
    insp.plot_frame_spectra(original_frame, bp)

    # Test and plot desmiling with INTR method
    desmiled_frame = smile.apply_shift_matrix(original_frame, shift_matrix, target_is_cube=False, method=1)
    insp.plot_frame(original_frame, sl_list, True, True, True)
    insp.plot_frame_spectra(original_frame, bp)


    # Spectral cubes
    # -------------
    # cube_sl_locations = [360,430,515,665,780,827,930,970,1025,1382,2060]
    # test_cube_name = 'checker001a'
    # reference_frame_name = 'checkerDesRef'
    # original_cube = xr.open_dataset(os.path.normpath('frames/' + test_cube_name + '.nc'))
    # reference_frame = xr.open_dataset(os.path.normpath('frames/' + reference_frame_name + '.nc'))
    
    # bp = smile.construct_bandpass_filter(reference_frame, cube_sl_locations, 30)
    # sl_list = smile.construct_spectral_lines(reference_frame, cube_sl_locations, bp)
    # shift_matrix = smile.construct_shift_matrix(sl_list, reference_frame.x.size,  reference_frame.y.size)

    # desmiled_cube = smile.apply_shift_matrix(original_cube, shift_matrix, target_is_cube=True, method=0)
    # insp.plot_frame(original_frame, sl_list, True, True, True)
    # insp.plot_frame_spectra(original_frame, bp)
