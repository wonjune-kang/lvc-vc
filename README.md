# LVC-VC

### Wonjune Kang, Mark Hasegawa-Johnson, Deb Roy

This repository contains code for LVC-VC, a zero-shot voice conversion model described in our Interspeech 2023 paper, [End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions](https://arxiv.org/abs/2205.09784), implemented in PyTorch.

Additionally, it includes code for a larger, improved version of our model (not described in the paper), which we call **LVC-VC XL**. This version of the model uses a larger channel size of 32 (rather than 16) in its LVC layers, utilizes embeddings from [XLSR-53](https://arxiv.org/abs/2006.13979) as content features, and uses information perturbation to extract only linguistic information from them (as done in [NANSY](https://arxiv.org/abs/2110.14513)). It also uses speaker embeddings from [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) rather than [Fast ResNet-34](https://arxiv.org/abs/2003.11982). LVC-VC XL achieves significantly better performance over the base version of our model in terms of both intelligibility and voice style transfer performance, and we encourage you to use it rather than the base version if memory and compute allow.

Audio samples are available on our [demo page](https://lvc-vc.github.io/lvc-vc-demo/).

If you find this work or our code useful, please consider citing our paper:

```
@inproceedings{kang23b_interspeech,
  author={Wonjune Kang and Mark Hasegawa-Johnson and Deb Roy},
  title={{End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2303--2307},
  doi={10.21437/Interspeech.2023-2298}
}
```

## Prerequisites

You can install all dependencies by running

```
pip install -r requirements.txt
```

## Pre-trained model weights

Create a directory called ```weights``` in the working directory, and save the pretrained weights from the Google Drive link. We include pre-trained weights for LVC-VC, Fast ResNet-34, LVC-VC XL, and ECAPA-TDNN.

**[Google Drive Link](https://drive.google.com/drive/folders/1ZaiJS-dXaTJnZbxuHV_sFB0IgZ42yS4F?usp=sharing)**

## Data preprocessing

If you want to train a model from scratch, you will need to download the [VCTK dataset](https://datashare.ed.ac.uk/handle/10283/3443). Then, run

```
./preprocess_data.sh
```

to preprocess all the data. This script will:

1. Resample all audio in VCTK from 48 kHz to 16 kHz
2. Extract F0 metadata (log F0 median and standard deviation) for each speaker and save them as a dictionary ```{speaker_id: {'median': --, 'std': --}}``` in pickle format
3. Extract spectrograms and normalized F0 contours (two sets of F0 contours will be extracted, matching the window and hops for spectrograms and XLSR-53 features)
4. Split the data into 99 seen and 10 unseen speakers, and then further split the seen speakers' utterances into train and test sets in a 9:1 ratio
5. Extract speaker embeddings from either Fast ResNet-34 or ECAPA-TDNN (depending on the config file specified), fit Gaussians for each speaker's embeddings, and save them as a dictionary ```{'speaker_id': sklearn.mixture.GaussianMixture object}``` in pickle format

**Note that the preprocessing scripts have directories and file paths hardcoded in. Therefore, you will need to go in and change them as needed if running on your own machine.** The script will also extract and preprocess data needed for both the base and XL versions of LVC-VC. If you are only interested in training one or the other, then comment out the corresponding parts of the code as needed.

## Training

You can train a model by specifying a config file, base GPU index, and run name. Note that the base GPU index specifies the first GPU to use on your machine, and then uses the next consecutive ```num_gpus``` specified in the config file (e.g. if you specify ```-g 0``` and ```num_gpus: 4```, then you will train using GPUs ```[0,1,2,3]```. You can also continue training from a checkpoint using the ```-p``` flag.

```
python3 trainer.py \
  -c config/config_wav2vec_ecapa_c32.yaml \
  -g 0 \
  -n lvc_vc_wav2vec_ecapa_c32
```

If you are training the base version of LVC-VC using spectrograms as content features, you will also need to supplement self-reconstructive training with speaker similarity criterion (SSC). To do this, first train a model to convergence using ```config/config_spect_c16.yaml```, and then continue training from the last checkpoint with ```config/config_spect_c16_ssc.yaml```. Training with SSC loss will save model checkpoints every 400 iterations; you may need to test a few checkpoints to find the optimal trade-off between audio quality and voice style transfer performance.

(This is one of the reasons why we encourage you to use LVC-VC XL; it achieves better performance without needing the additional SSC loss step.)

## Inference

Depending on which version of the model you are using, run either ```inference_wav2vec.py``` or ```inference_spect.py```. If you are running ```inference_wav2vec.py``` without having run the data preprocessing first, you can use the metadata pickle files in the ```metadata``` directory of this repository.

```
python3 inference_wav2vec.py \
  -c config/config_wav2vec_ecapa_c32.yaml \
  -p weights/lvc_vc_xl_vctk.pt \
  -e weights/ecapa_tdnn_pretrained.pt \
  -g 0 \
  -s source_utterance_file \
  -t target_utterance_file \
  -o output_file_name
```

```
python3 inference_spect.py \
  -c config/config_spect_c16.yaml \
  -p weights/lvc_vc_vctk.pt \
  -e weights/resnet34sel_pretrained.pt \
  -g 0 \
  -s source_utterance_file \
  -t target_utterance_file \
  -o output_file_name
```

## References

We referred to the following repositories and resources in our code:

- https://github.com/mindslab-ai/univnet for overall repository structure, and for the core modules for the model architecture and training
- https://github.com/clovaai/voxceleb_trainer for the implementation and pre-trained weights for Fast ResNet-34
- https://github.com/TaoRuijie/ECAPA-TDNN for the implementation and pre-trained weights for ECAPA-TDNN (used for LVC-VC XL)
- https://github.com/dhchoi99/NANSY for the information perturbation functions that use parselmouth in ```utils/perturbations.py``` (used for LVC-VC XL)
- https://huggingface.co/facebook/wav2vec2-large-xlsr-53 for the XLSR-53 weights (used for LVC-VC XL)
