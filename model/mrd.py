import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorR(torch.nn.Module):
    def __init__(self, hp, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.sampling_rate = hp.audio.sampling_rate
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        norm_f = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        # Compute magnitude spectrogram of input signal.
        x = self.spectrogram(x)
        x = x.unsqueeze(1)

        # Pass spectrogram through convolutional discriminator network.
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)

        # fmap is list of discriminator's layer-wise feature map outputs
        # (for feature matching loss in HiFi-GAN; unused in UnivNet).
        fmap.append(x)
    
        # x is 1D tensor of frame-wise discriminator scores (0-1).
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        # Convert window length and hop length from ms to samples.
        hop_length_ms, win_length_ms = self.resolution
        hop_length = int(0.001 * hop_length_ms * self.sampling_rate)
        win_length = int(0.001 * win_length_ms * self.sampling_rate)

        # Compute n_fft based on window length samples.
        n_fft = int(math.pow(2, int(math.log2(win_length)) + 1))

        # Pad edges of signal to center FFT shift frames at start and end.
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)

        # Compute STFT and convert to magnitude spectrum.
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, center=False) #[B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, hp):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
