import xarray as xr
import os

import smile_correction as smile
import inspector as insp

if __name__ == '__main__':

    test_frame_name = 'test_frame'
    test_frame_sl_locations = [230,267,319,369,439,525,635,681,733,793,840,978,1030,1212,1400]
    original_frame = xr.open_dataset(os.path.normpath('frames/' + test_frame_name + '.nc')).frame
    # insp.plot_frame(original_frame, None, True, True, True)
    bp = smile.construct_bandpass_filter(original_frame, test_frame_sl_locations, 30)
    sl_list = smile.construct_spectral_lines(original_frame, test_frame_sl_locations, bp)
    shift_matrix = smile.construct_shift_matrix(sl_list, original_frame.x.size,  original_frame.y.size)

    # desmiled_frame = smile.apply_shift_matrix(original_frame, shift_matrix, target_is_cube=False)
    # insp.plot_frame(original_frame, sl_list, True, True, True)
    # insp.plot_frame_spectra(original_frame, bp)

    desmiled_frame = smile.apply_shift_matrix(original_frame, shift_matrix, target_is_cube=False, method=1)
    insp.plot_frame(original_frame, sl_list, True, True, True)
    insp.plot_frame_spectra(original_frame, bp)