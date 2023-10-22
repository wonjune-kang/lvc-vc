import itertools
import logging
import math
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2ForPreTraining

from dataset import create_dataloader
from model.discriminator import Discriminator
from model.generator import Generator
from model.ResNetBlocks import SEBasicBlock
from model.ResNetSE34L import ResNetSE
from model.ssc import SpeakerSimilarityCriterion
from utils.stft import TacotronSTFT
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.utils import get_commit_hash
from utils.writer import MyWriter
from validation import validate


def train(args, hp, hp_str):

    init_epoch = -1
    step = 0

    # Initialize logger and writer.
    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(hp, log_dir)

    # Set device (CUDA or CPU).
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Create train and validation dataloaders.
    trainloader = create_dataloader(hp, True)
    valloader = create_dataloader(hp, False)

    # Create STFT computation object.
    stft = TacotronSTFT(
        filter_length=hp.audio.filter_length,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mel_channels,
        sampling_rate=hp.audio.sampling_rate,
        mel_fmin=hp.audio.mel_fmin,
        mel_fmax=hp.audio.mel_fmax,
        center=False,
        device=device
    )

    # Initialize wav2vec 2.0 model for linguistic feature extraction.
    if hp.train.use_wav2vec:
        wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
        for param in wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        print("Loaded wav2vec 2.0.\n")
    else:
        wav2vec2 = None

    # Initialize models and optimizers.
    model_g = Generator(hp)
    model_d = Discriminator(hp)

    optim_g = torch.optim.AdamW(
        model_g.parameters(),
        lr=hp.train.adam.lr,
        betas=(hp.train.adam.beta1, hp.train.adam.beta2)
    )
    optim_d = torch.optim.AdamW(
        model_d.parameters(),
        lr=hp.train.adam.lr,
        betas=(hp.train.adam.beta1, hp.train.adam.beta2)
    )

    # Load model weights from checkpoint if specified.
    githash = get_commit_hash()
    chkpt_path = args.checkpoint_path
    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location=device)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        # If training on GPU and loading optimizer state_dict, manually move
        # parameters to GPU.
        if device == torch.device(f"cuda:{args.gpu_idx}"):
            for state in optim_g.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(args.gpu_idx)
            for state in optim_d.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(args.gpu_idx)

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # Parallelize across multiple GPUs if specified.
    torch.cuda.manual_seed(hp.train.seed)
    if device == torch.device(f"cuda:{args.gpu_idx}"):
        if hp.train.num_gpus > 1:
            assert torch.cuda.device_count() >= hp.train.num_gpus
            model_g = nn.DataParallel(
                model_g,
                device_ids=list(range(args.gpu_idx, args.gpu_idx+hp.train.num_gpus))
            )
            model_d = nn.DataParallel(
                model_d,
                device_ids=list(range(args.gpu_idx, args.gpu_idx+hp.train.num_gpus))
            )
            if hp.train.use_wav2vec:
                wav2vec2 = nn.DataParallel(
                    wav2vec2,
                    device_ids=list(range(args.gpu_idx, args.gpu_idx+hp.train.num_gpus))
                )

    # Send models to device.
    model_g = model_g.to(device)
    model_d = model_d.to(device)
    if hp.train.use_wav2vec:
        wav2vec2 = wav2vec2.to(device)

    if hp.train.use_ssc:
        speaker_encoder = ResNetSE(
            SEBasicBlock,
            hp.ssc.se.layers,
            hp.ssc.se.num_filters,
            hp.ssc.se.spk_emb_dim
        )
        speaker_encoder.load_pretrained(hp.ssc.se.pretrained_weight_path)
        print("Loaded pretrained ResNet speaker encoder.")

        ssc_criterion = SpeakerSimilarityCriterion(speaker_encoder, device)
        ssc_criterion.to(device)

    # This accelerates training when the size of minibatch is always consistent.
    # If not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    ############
    # Training #
    ############
    start_step = step

    model_g.train()
    model_d.train()

    # Convert window length and hop length from ms to samples.
    resolutions = []
    for hop_length_ms, win_length_ms in eval(hp.mrd.resolutions):
        hop_length = int(0.001 * hop_length_ms * hp.audio.sampling_rate)
        win_length = int(0.001 * win_length_ms * hp.audio.sampling_rate)
        n_fft = int(math.pow(2, int(math.log2(win_length)) + 1))
        resolutions.append((n_fft, hop_length, win_length))

    stft_criterion = MultiResolutionSTFTLoss(device, resolutions)

    # Training loop.
    for epoch in itertools.count(init_epoch+1):

        if hp.train.use_ssc and epoch > init_epoch + hp.ssc.finetune_epochs:
            print("SSC finetuning complete.")
            break

        loader = tqdm(trainloader, desc='Loading train data')

        for audios, perturbed_audios, spects, speaker_feats, f0_norms in loader:

            # First item of each list of features is used for reconstruction.
            audio = audios[0].to(device)
            speaker_feat = speaker_feats[0].to(device)
            f0_norm = f0_norms[0].to(device)

            # Extract wav2vec 2.0 features from perturbed audio.
            if hp.train.use_wav2vec:
                perturbed_audio = perturbed_audios[0].to(device)
                with torch.no_grad():
                    wav2vec2_outputs = wav2vec2(perturbed_audio, output_hidden_states=True)
                feat = wav2vec2_outputs.hidden_states[12]  # (B, N, 1024)
                feat = feat.permute((0, 2, 1))  # (B, 1024, N)

            # Or use low-quefrency liftered mel spectrogram.
            else:
                feat = spects[0].to(device)  # (B, 80, N)

            #############
            # Generator #
            #############
            optim_g.zero_grad()

            # Perform self-reconstruction of audio.
            noise = torch.randn(hp.train.batch_size, hp.gen.noise_dim, feat.size(2)).to(device)
            content_feature = torch.cat((feat, f0_norm), dim=1)
            fake_audio = model_g(content_feature, noise, speaker_feat)

            # Crop all audio to be the length of the shortest signal (in case of
            # slight mismatches when upsampling input noise sequence).
            min_size = min(audio.size(2), fake_audio.size(2))
            audio = audio[:, :, :min_size]
            fake_audio = fake_audio[:, :, :min_size]

            # Compute Multi-Resolution STFT Loss.
            # (spectral convergence loss + log STFT magnitude loss)
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))

            stft_lamb = hp.train.stft_lamb
            stft_loss = (sc_loss + mag_loss) * stft_lamb

            # MRD, MPD losses.
            res_fake, period_fake = model_d(fake_audio)

            # Compute LSGAN loss for all frames.
            score_loss = 0.0
            for (_, score_fake) in res_fake + period_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

            # Average across frames.
            score_loss = score_loss / len(res_fake + period_fake)

            # Overall generator loss (L_G).
            loss_g = score_loss + stft_loss

            # We only use SSC loss if training using low-quefrency liftered
            # spectrograms.
            if hp.train.use_ssc:
                # Use the last utterance in the list as the target speaker.
                vc_target_speaker_feat = speaker_feats[-1].to(device)

                src_audios = [src_audio.squeeze().to(device) for src_audio in audios[:-1]]
                fake_ssc_audios = []
                for i in range(len(spects)-1):
                    source_spect = spects[i].to(device)
                    source_f0_norm = f0_norms[i].to(device)

                    content_feature = torch.cat((source_spect, source_f0_norm), dim=1)
                    noise = torch.randn(
                        hp.train.batch_size,
                        hp.gen.noise_dim,
                        source_spect.size(2)
                    ).to(device)

                    fake_ssc_audio = model_g(
                        content_feature,
                        noise,
                        vc_target_speaker_feat
                    ).squeeze(1)
                    fake_ssc_audios.append(fake_ssc_audio)

                pos_ssc_loss, neg_ssc_loss = ssc_criterion(
                    fake_ssc_audios,
                    src_audios,
                    vc_target_speaker_feat[:, :hp.ssc.se.spk_emb_dim]
                )

                pos_ssc_lamb = min(
                    hp.ssc.pos_ssc_lamb,
                    hp.ssc.pos_ssc_lamb * ((step-start_step) / hp.ssc.ssc_annealing_step)
                )  # len(loader)
                neg_ssc_lamb = min(
                    hp.ssc.neg_ssc_lamb,
                    hp.ssc.neg_ssc_lamb * ((step-start_step) / hp.ssc.ssc_annealing_step)
                )  # len(loader)
                ssc_loss = (pos_ssc_lamb * pos_ssc_loss) + (neg_ssc_lamb * neg_ssc_loss)

                loss_g += ssc_loss

            loss_g.backward()
            optim_g.step()

            #################
            # Discriminator #
            #################
            optim_d.zero_grad()

            # MRD, MPD losses.
            res_fake, period_fake = model_d(fake_audio.detach())  # fake audio from generator
            res_real, period_real = model_d(audio)  # real audio

            # Compute LSGAN loss for all frames.
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(
                res_fake + period_fake, res_real + period_real
            ):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))

            # Compute average to get overall discriminator loss (L_D).
            loss_d = loss_d / len(res_fake + period_fake)

            loss_d.backward()
            optim_d.step()

            step += 1

            ###########
            # Logging #
            ###########
            loss_g = loss_g.item()
            loss_d = loss_d.item()
            stft_loss = stft_loss.item()

            if hp.train.use_ssc:
                ssc_loss = ssc_loss.item()
                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d,
                                        stft_loss, score_loss,
                                        step, ssc_loss=ssc_loss)
                    loader.set_description(
                        "g %.04f d %.04f ssc %.04f | step %d" % (loss_g, loss_d, ssc_loss, step)
                    )

            else:
                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d,
                                        stft_loss, score_loss, step)
                    loader.set_description("g %.04f d %.04f | step %d" % (loss_g, loss_d, step))

            if hp.train.use_ssc and step % hp.log.ssc_validation_interval_steps == 0:
                with torch.no_grad():
                    validate(
                        hp,
                        model_g,
                        model_d,
                        wav2vec2,
                        valloader,
                        stft,
                        writer,
                        step,
                        device,
                        ssc=True
                    )

                save_path = os.path.join(pt_dir, '%s_%04d_%d.pt' % (args.name, epoch, step))
                torch.save({
                    'model_g': (model_g.module if hp.train.num_gpus > 1 else model_g).state_dict(),
                    'model_d': (model_d.module if hp.train.num_gpus > 1 else model_d).state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

        ##############
        # Validation #
        ##############
        if epoch % hp.log.validation_interval == 0:
            with torch.no_grad():
                validate(hp, model_g, model_d, wav2vec2, valloader, stft, writer, step, device)

        # Save model weights to checkpoint.
        if epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pt_dir, '%s_%04d.pt'
                                     % (args.name, epoch))
            torch.save({
                'model_g': (model_g.module if hp.train.num_gpus > 1 else model_g).state_dict(),
                'model_d': (model_d.module if hp.train.num_gpus > 1 else model_d).state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                'githash': githash,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
