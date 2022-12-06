#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .ResNetBlocks import *


class ResNetSE(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_filters,
                 embedding_dim,
                 encoder_type='SAP',
                 n_mels=40,
                 log_input=True):

        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(embedding_dim, encoder_type))
        
        # Model specifications and hyperparameters.
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input

        # Input convolutional layer.
        self.conv1 = nn.Conv2d(1,
                               num_filters[0],
                               kernel_size=7,
                               stride=(2, 1),
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        # Convolutional residual block layers.
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        # Mel spectrogram transformer.
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                            n_fft=512,
                                                            win_length=400,
                                                            hop_length=160,
                                                            window_fn=torch.hamming_window,
                                                            n_mels=n_mels)

        # Instance normalization per mel channel in spectrogram.
        self.instancenorm = nn.InstanceNorm1d(n_mels)

        # Self-attentive pooling.
        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        # Attentive statistics pooling.
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')

        # Output fully connected layer
        self.fc = nn.Linear(out_dim, embedding_dim)

        # Initialize weights for Conv2d and batch norm layers in network.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """
        Create layer (stack) of residual blocks with a given planes (channel) size.
        """
        # Add Conv2d layer for downsampling to residual blocks when channel
        # dimension changes or stride of residual blocks is greater than 1.
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # Add convolutional residual blocks.
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        """
        For making encoder attention parameter.
        """
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def load_pretrained(self, weight_path):
        checkpoint = torch.load(weight_path)

        new_state_dict = {}
        for k, v in checkpoint.items():
            try:
                new_state_dict[k[6:]] = checkpoint[k]
            except:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)

    def forward(self, x):
        # Compute mel spectrogram from time domain input signal.
        # with torch.no_grad():
        # with torch.cuda.amp.autocast(enabled=False):
        x = self.torchfb(x) + 1e-6
        if self.log_input: # mel spectrogram -> log mel spectrogram
            x = x.log()
        x = self.instancenorm(x).unsqueeze(1)#.detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = torch.mean(x, dim=2, keepdim=True)

        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        
        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def MainModel(embedding_dim=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, embedding_dim)
    return model
