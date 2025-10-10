import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, sys

# Add the project root to Python path
project_root = '/exports/lkeb-hpc/xwan/osteosarcoma/working/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.pytorch_gaussian_process import  pyRandomFeatureGaussianProcess, mean_field_logits  # Custom
from models.pytorch_metrics_dev import Metrics  # Custom
from models.pytorch_spectral_normalization import SpectralNormalizationConv3D # Custom

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, width_multiplier, stride=1, use_SN=False, spec_norm_bound=None):
        self.width_multiplier = width_multiplier
        super(BasicBlock, self).__init__()
        self.snb = spec_norm_bound if use_SN else None
            
        if use_SN:
            self.conv1 = SpectralNormalizationConv3D(
                nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                spec_norm_bound=self.snb)
            self.conv2 = SpectralNormalizationConv3D(
                nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                spec_norm_bound=self.snb)
        else:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if use_SN:
                self.shortcut = nn.Sequential(
                SpectralNormalizationConv3D(nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                            spec_norm_bound=self.snb),
                nn.BatchNorm3d(planes)
                )
            else:
                self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual Connect
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width_multiplier, use_SN = False, use_GP = False, drate=0.3, num_classes=10, snb=None, k_s=None):
        super(ResNet, self).__init__()
        print('>> WideResNet-{}-{}:Initialization | SN:{} | GP:{} '.format(num_blocks[0] * 6 + 4,
                                                          width_multiplier, int(use_SN), int(use_GP)))
        self.width_multiplier = width_multiplier
        self.in_planes = 16
        self.use_GP = use_GP
        self.use_SN = use_SN
        self.dropout = nn.Dropout(drate)
        self.snb = snb if use_SN else None
        self.k_s = k_s if use_GP else None

        if self.use_SN:  # For MRI data, in_channels=1; For CIFAR, in_channels=3
            self.conv1 = SpectralNormalizationConv3D(nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                                     spec_norm_bound=snb)
        else:
            self.conv1 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm3d(16)
        self.layer1 = self._make_layer(block, 16 * self.width_multiplier, num_blocks[0], stride=1, use_SN=use_SN, spec_norm_bound=self.snb)
        self.layer2 = self._make_layer(block, 32 * self.width_multiplier, num_blocks[1], stride=2, use_SN=use_SN, spec_norm_bound=self.snb)
        self.layer3 = self._make_layer(block, 64 * self.width_multiplier, num_blocks[2], stride=2, use_SN=False, spec_norm_bound=None)  # No SN in the last layer

        if self.use_GP:
            self.projection = nn.Linear(
                              in_features=64 * self.width_multiplier,
                              out_features=64,    # 128
                              bias=False)
            nn.init.normal_(self.projection.weight, mean=0.0, std=1.0)
            self.projection.weight.requires_grad = False

            self.classifier = pyRandomFeatureGaussianProcess(
                in_features=64,
                out_features=num_classes,
                gp_cov_momentum=0.999,
                gp_kernel_scale=self.k_s,
                gp_cov_ridge_penalty=0.001
            )
        else:
            self.classifier = nn.Linear(in_features=64 * self.width_multiplier,
                                        out_features=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_SN=False, spec_norm_bound=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in range(len(strides)):
            if s != 0:
                self.in_planes = planes
            stride = strides[s]
            layers.append(block(self.in_planes, planes, self.width_multiplier, stride, use_SN=use_SN, spec_norm_bound=spec_norm_bound))
        return nn.Sequential(*layers)

    def forward(self, x, return_covmat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool3d(out, kernel_size=out.shape[2:])  # CIFAR: out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)  # Flatten 1x1x640 --> 640 ; required by a fully-connected layer later
        out = self.dropout(out)

        if self.use_GP:
            out = self.projection(out)
            logits, covmat = self.classifier(out)
        else:
            logits = self.classifier(out)
        if return_covmat:
            return logits, covmat
        return logits