import tqdm
import torch
import torch.nn.functional as F


def validate(hp, generator, discriminator,
             valloader, stft, writer, step, device, ssc=False):

    # Set models to evaluation mode.
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (audios, spects, speaker_feats, f0_norms) in enumerate(loader):
        
        if ssc and idx == hp.log.num_audio:
            return

        spect, _ = spects
        speaker_feat, speaker_feat2 = speaker_feats
        audio, audio2 = audios
        f0_norm, _ = f0_norms

        # Mel spectrogram, audio signal, F0 contour, and speaker features.
        spect = spect.to(device)
        audio = audio.to(device)
        f0_norm = f0_norm.to(device)
        speaker_feat = speaker_feat.to(device)
        speaker_feat2 = speaker_feat2.to(device)

        # Generate fake audio for reconstruction.
        noise = torch.randn(1, hp.gen.noise_dim, spect.size(2)).to(device)
        content_feature = torch.cat((spect, f0_norm), dim=1)

        recon_audio = generator(content_feature, noise, speaker_feat)

        # Generate voice converted audio.
        noise = torch.randn(1, hp.gen.noise_dim, spect.size(2)).to(device)
        vc_audio = generator(content_feature, noise, speaker_feat2)

        recon_audio = recon_audio[:,:,:audio.size(2)]
        vc_audio = vc_audio[:,:,:audio.size(2)]

        # Compute mel spectrograms for reconstructed (fake) and real audio.
        spect_fake = stft.mel_spectrogram(recon_audio.squeeze(1))
        spect_real = stft.mel_spectrogram(audio.squeeze(1))

        # Compute L1 loss for spect spectrograms.
        mel_loss += F.l1_loss(spect_fake, spect_real).item()

        # Save generated audio and spectrogram figures to Tensorboard.
        if idx < hp.log.num_audio:
            spec_fake = stft.linear_spectrogram(recon_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio = audio[0][0].cpu().detach().numpy()
            recon_audio = recon_audio[0][0].cpu().detach().numpy()
            vc_audio = vc_audio[0][0].cpu().detach().numpy()
            spec_fake = spec_fake[0].cpu().detach().numpy()
            spec_real = spec_real[0].cpu().detach().numpy()
            writer.log_fig_audio(audio, recon_audio, spec_fake, spec_real, idx, step)
            writer.log_fig_vc_audio(audio, audio2, vc_audio, idx, step)
            
    # Compute and write average spect loss.
    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    torch.backends.cudnn.benchmark = True
