import numpy as np
import scipy


class FrequencyWarp():
    
    def __init__(self, compress_overlap_bins=2):
        self.overlap_bins = compress_overlap_bins

    def compress_spectrum_frequency(self, spect, compress_ratio=0.9):
        x = np.linspace(0, spect.shape[0]-1, spect.shape[0])
        x_new = np.linspace(0, spect.shape[0]-1, int(spect.shape[0]*compress_ratio))

        end_idx = int(spect.shape[0]*compress_ratio)
        # num_zeros = spect.shape[0] - end_idx + self.overlap_bins
        
        y_new = np.zeros(spect.shape)
        for i in range(spect.shape[1]):
            y = spect[:,i]
            f = scipy.interpolate.interp1d(x, y, kind='cubic')            
            y_new[:end_idx, i] = f(x_new)
        
        last_bins = spect[-10:,:]
        filler_shape = y_new[end_idx-self.overlap_bins:,:].shape
        noise = np.random.normal(np.mean(last_bins), np.std(last_bins), filler_shape)
        
        y_new[end_idx-self.overlap_bins:,:] = noise

        return y_new

    def expand_spectrum_frequency(self, spect, expand_ratio=1.1):
        x = np.linspace(0, spect.shape[0]-1, spect.shape[0])
        x_new = np.linspace(0, spect.shape[0]-1, int(spect.shape[0]*expand_ratio))[:spect.shape[0]]

        y_new = np.zeros(spect.shape)
        for i in range(spect.shape[1]):
            y = spect[:,i]
            f = scipy.interpolate.interp1d(x, y, kind='cubic')
            y_new[:int(spect.shape[0]), i] = f(x_new)

        return y_new

    def warp_spect_frequency(self, spect, warp_ratio):
        if warp_ratio == 1.0:
            return spect
        elif warp_ratio > 1.0:
            warped_spectrum = self.expand_spectrum_frequency(spect, warp_ratio)
        else:
            warped_spectrum = self.compress_spectrum_frequency(spect, warp_ratio)

        return warped_spectrum.astype(np.float32)