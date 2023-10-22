import torch
import torch.nn.functional as F
import tqdm


def validate(hp, generator, discriminator, wav2vec2, valloader, stft,
             writer, step, device, ssc=False):

    # Set models to evaluation mode.
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (audios, _, spects, speaker_feats, f0_norms) in enumerate(loader):

        if ssc and idx == hp.log.num_audio:
            return

        audio, audio2 = audios
        speaker_feat, speaker_feat2 = speaker_feats
        f0_norm, _ = f0_norms

        # Mel spectrogram, audio signal, F0 contour, and speaker features.
        audio = audio.to(device)
        f0_norm = f0_norm.to(device)
        speaker_feat = speaker_feat.to(device)
        speaker_feat2 = speaker_feat2.to(device)

        # Extract wav2vec 2.0 features from audio.
        if hp.train.use_wav2vec:
            wav2vec2_outputs = wav2vec2(audio.squeeze(1), output_hidden_states=True)
            feat = wav2vec2_outputs.hidden_states[12]  # (B, N, 1024)
            feat = feat.permute((0, 2, 1))  # (B, 1024, N)

            # Crop feat or f0_norm to match shorter feature (they may be
            # off by 1 or 2 because of slight mismatches in frame-wise processing).
            min_feat_len = min(feat.size(2), f0_norm.size(2))
            feat = feat[:, :, :min_feat_len]
            f0_norm = f0_norm[:, :, :min_feat_len]

        # Or use low-quefrency liftered mel spectrogram.
        else:
            feat, _ = spects
            feat = feat.to(device)

        # Generate fake audio for reconstruction.
        noise = torch.randn(1, hp.gen.noise_dim, feat.size(2)).to(device)
        content_feature = torch.cat((feat, f0_norm), dim=1)

        recon_audio = generator(content_feature, noise, speaker_feat)

        # Generate voice converted audio.
        noise = torch.randn(1, hp.gen.noise_dim, feat.size(2)).to(device)
        vc_audio = generator(content_feature, noise, speaker_feat2)

        # Crop all audio to be the length of the shortest signal (in case of
        # slight mismatches when upsampling input noise sequence).
        min_audio_len = min(audio.size(2), recon_audio.size(2))
        audio = audio[:, :, :min_audio_len]
        recon_audio = recon_audio[:, :, :min_audio_len]
        vc_audio = vc_audio[:, :, :min_audio_len]

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
