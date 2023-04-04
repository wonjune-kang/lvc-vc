import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .lvcnet import LVCBlock

MAX_WAV_VALUE = 32768.0

class Generator(nn.Module):
    """LVC-VC Generator (based on UnivNet)"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        # Content feature is low-quefrency liftered mel spectrogram stacked
        # with normalized F0 contour.
        self.content_feat_dim = hp.audio.feat_dim + hp.audio.f0_norm_dim

        # Speaker feature is speaker embedding stacked with speaker's
        # quantized average f0 median.
        self.speaker_feat_dim = hp.audio.spk_emb_dim + hp.audio.spk_quant_f0_dim

        # Input hyperparameters.
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size

        # Define residual LVC block stack.
        self.res_stack = nn.ModuleList()

        # Start with initial hop value 1 and multiple by values in
        # hp.gen.strides at each LVC block.
        hop_length = 1
        for stride in hp.gen.strides:
            # Define LVC block that dilates input noise by stride.
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    self.content_feat_dim,
                    stride=stride,
                    speaker_emb_dim=self.speaker_feat_dim,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )
        
        # Initial convolution layer to Generator.
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(
                hp.gen.noise_dim,
                channel_size,
                7,
                padding=3,
                padding_mode='reflect'
            )
        )

        # Final convolution block after LVC block stack.
        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.utils.weight_norm(
                nn.Conv1d(
                    channel_size,
                    1,
                    7,
                    padding=3,
                    padding_mode='reflect'
                )
            ),
            nn.Tanh(),
        )

    def forward(self, c, z, s):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram
                        (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        # Initial convolutional layer: (B, c_g, L)
        z = self.conv_pre(z)

        # Pass through LVC residual stack.
        for res_block in self.res_stack:
            res_block.to(z.device)

            # (B, c_g, L * s_0 * ... * s_i)
            z = res_block(z, c, s)

        # Final convolutional layer: (B, 1, L * 256)
        z = self.conv_post(z)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # Don't remove weight norm while validation in training loop.
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    # def inference(self, c, z, s):
    #     # Pad input mel with zeros to cut artifact at end of generated signal
    #     # (see https://github.com/seungwonpark/melgan/issues/8).
    #     zero = torch.full((1, self.content_feat_dim, 10), -11.5129).to(c.device)
    #     mel = torch.cat((c, zero), dim=2)
        
    #     # Input noise sequence.
    #     if z is None:
    #         z = torch.randn(1, self.noise_dim, mel.size(2)).to(mel.device)

    #     # Generate audio.
    #     audio = self.forward(mel, z, s)
    #     audio = audio.squeeze() # collapse all dimension except time axis

    #     # Cut samples corresponding to earlier padding.
    #     audio = audio[:-(self.hop_length*10)]

    #     # Convert audio tensor to int16.
    #     audio = MAX_WAV_VALUE * audio
    #     audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
    #     audio = audio.short()

    #     return audio

if __name__ == '__main__':
    hp = OmegaConf.load('../config/config.yaml')
    model = Generator(hp)

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
