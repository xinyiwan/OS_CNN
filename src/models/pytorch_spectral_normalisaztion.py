"""
Implements a spectral normalization for the weights in PyTorch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormWithBound(nn.Module):
    def __init__(self, module, spec_norm_bound=0.95, n_power_iterations=1, eps=1e-12):
        super(SpectralNormWithBound, self).__init__()
        self.module = module
        self.spec_norm_bound = spec_norm_bound
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.u = None
        self.v = None

    def update_weights(self):
        w = self.module.weight
        w_mat = w.view(w.size(0), -1)  # [out_features, in_features]
        if self.u is None:
            self.u = F.normalize(torch.randn(1, w_mat.size(0)), dim=1, eps=self.eps)
            self.v = F.normalize(torch.randn(1, w_mat.size(1)), dim=1, eps=self.eps)

        for _ in range(self.n_power_iterations):
            self.v = F.normalize(torch.matmul(self.u, w_mat.t()), p=2, dim=1, eps=self.eps)
            self.u = F.normalize(torch.matmul(self.v, w_mat), p=2, dim=1, eps=self.eps)

        sigma = torch.matmul(torch.matmul(self.v, w_mat), self.u.t()).squeeze()
        factor = min(self.spec_norm_bound / sigma.item(), 1.0)

        w_norm = w * factor
        return w_norm

    def forward(self, x):
        w_norm = self.update_weights()
        if isinstance(self.module, nn.Linear):
            output = F.linear(x, w_norm, self.module.bias)
        else:
            raise NotImplementedError("SpectralNormWithBound only supports nn.Linear modules.")
        return output

class SpectralNormalizationConv2D(nn.Module):
    def __init__(self, module, spec_norm_bound=8, n_power_iterations=1, eps=1e-12, legacy_mode=False):
        super(SpectralNormalizationConv2D, self).__init__()
        self.legacy_mode = legacy_mode
        self.module = module
        self.spec_norm_bound = spec_norm_bound
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.u = None
        self.v = None
        self.w = self.module.weight

    def update_weights(self, x):
        padding = [math.floor((s - 1) / 2) for s in self.module.kernel_size]
        self.w_shape = self.w.shape
        self.strides = self.module.stride
        self.batchsize = x.shape[0]
        self.uv_dim = self.batch_size if self.legacy_mode else 1

        # Resolve shapes
        self.in_height = x.shape[2]
        self.in_width = x.shape[3]
        self.in_channel = self.w_shape[1]

        self.out_height = self.in_height // self.strides[0]
        self.out_width = self.in_width // self.strides[1]
        self.out_channel = self.w_shape[0]

        # TF: [batch, h, w, c]
        # PY: [batch, c, h, w]
        self.in_shape = (self.uv_dim, self.in_channel, self.in_height, self.in_width)
        self.out_shape = (self.uv_dim, self.out_channel, self.out_height, self.out_width)
        self.w_mat = self.w.view(self.w.shape[0], -1)

        if self.u is None or self.v is None:
            # Initialize u and v for power iteration
            self.u = F.normalize(torch.randn(self.out_shape), dim=1, eps=self.eps)
            self.v = F.normalize(torch.randn(self.in_shape), dim=1, eps=self.eps)

        u_hat = self.u.to(self.w.device)
        v_hat = self.v.to(self.w.device)
       
        #print("u-device:",self.u.device)
        #print("v-device:", self.v.device)
        #print("w-device:", self.w.device)

        for _ in range(self.n_power_iterations):
            # Updates v
            v_ = F.conv_transpose2d(u_hat, self.w, stride=self.strides, padding=padding)
            v_hat = F.normalize(v_, dim=1, eps=self.eps)
            v_hat = torch.reshape(v_hat, v_.shape)

            # Updates u
            u_ = F.conv2d(v_hat, self.w, stride=self.strides, padding=padding)
            u_hat = F.normalize(u_, dim=1, eps=self.eps)
            u_hat = torch.reshape(u_hat, u_.shape)

        # compute sigma
        v_w_hat = F.conv2d(v_hat, self.w, stride=self.strides, padding=padding)
        sigma = torch.matmul(torch.flatten(v_w_hat, 1), torch.flatten(u_hat, 1).t()).squeeze()
       #print("sigma", sigma)
        factor = min(self.spec_norm_bound / sigma, 1.0)
        # scale weights
        w_norm = self.w * factor
        return w_norm

    def forward(self, x):
        w_norm = self.update_weights(x)
        if isinstance(self.module, nn.Conv2d):
            output = F.conv2d(x, w_norm, bias=self.module.bias, stride=self.module.stride, padding=self.module.padding)
            #self.module.weight = torch.nn.Parameter(w_norm)
            #output = self.module(x)
            #self.module.weight = self.w
        else:
            raise NotImplementedError("SpectralNormWithBound only supports nn.Conv2d modules.")
        return output

class SpectralNormalizationConv3D(nn.Module):
    def __init__(self, module, spec_norm_bound=0.95, n_power_iterations=1, eps=1e-12, legacy_mode=False):
        super(SpectralNormalizationConv3D, self).__init__()
        self.legacy_mode = legacy_mode
        self.module = module
        self.spec_norm_bound = spec_norm_bound
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.u = None
        self.v = None
        self.w = self.module.weight

    def update_weights(self, x):
        #print("x.shape", x.shape)
        padding = [math.floor((s - 1) / 2) for s in self.module.kernel_size]
        self.w_shape = self.w.shape
        self.strides = self.module.stride
        self.batchsize = x.shape[0]
        self.uv_dim = self.batch_size if self.legacy_mode else 1

        # Resolve shapes
        self.in_height = x.shape[2]
        self.in_width = x.shape[3]
        self.in_z = x.shape[4]
        self.in_channel = self.w_shape[1]

        self.out_height = self.in_height // self.strides[0]
        self.out_width = self.in_width // self.strides[1]
        self.out_z = self.in_z // self.strides[2]
        self.out_channel = self.w_shape[0]

        # TF: [batch, h, w, c]
        # PY: [batch, c, h, w]
        self.in_shape = (self.uv_dim, self.in_channel, self.in_height, self.in_width, self.in_z)
        self.out_shape = (self.uv_dim, self.out_channel, self.out_height, self.out_width, self.out_z)
        #self.w_mat = self.w.view(self.w.shape[0], -1)

        if self.u is None or self.v is None:
            # Initialize u and v for power iteration
            self.u = F.normalize(torch.randn(self.out_shape), dim=1, eps=self.eps)
            self.v = F.normalize(torch.randn(self.in_shape), dim=1, eps=self.eps)

        u_hat = self.u.to(self.w.device)
        v_hat = self.v.to(self.w.device)

        for _ in range(self.n_power_iterations):
            # Updates v
            v_ = F.conv_transpose3d(u_hat, self.w, stride=self.strides, padding=padding)
            #print("v_shape", v_.shape)
            v_hat = v_.view(v_.size(0), -1)
            v_hat = F.normalize(v_hat, dim=1, eps=self.eps)
#            v_hat = F.normalize(v_, dim=1, eps=self.eps)
            v_hat = torch.reshape(v_hat, v_.shape)
            # Updates u
            u_ = F.conv3d(v_hat, self.w, stride=self.strides, padding=padding)
            #print("u_shape", u_.shape)
            u_hat = u_.view(u_.size(0),-1)
            u_hat = F.normalize(u_hat, dim=1, eps=self.eps)
#            u_hat = F.normalize(u_, dim=1, eps=self.eps)
            u_hat = torch.reshape(u_hat, u_.shape)

        # compute sigma
        v_w_hat = F.conv3d(v_hat, self.w, stride=self.strides, padding=padding)
        sigma = torch.matmul(torch.flatten(v_w_hat, 1), torch.flatten(u_hat, 1).t()).squeeze()
        #print("sigma:", sigma)
#        factor = min(self.spec_norm_bound / sigma.item(), 1.0)
        factor = min(self.spec_norm_bound / sigma, 1.0)
        # scale weights
        w_norm = self.w * factor
        return w_norm

    def forward(self, x):
        #print("self.w", self.w)
        #print("self.w", self.w.mean().item())

        w_norm = self.update_weights(x)
        if isinstance(self.module, nn.Conv3d):
            output = F.conv3d(x, w_norm, bias=self.module.bias, stride=self.module.stride, padding=self.module.padding)
            #self.module.weight = torch.nn.Parameter(w_norm)
            #output = self.module(x)
            #self.module.weight = self.w
        else:
            raise NotImplementedError("SpectralNormWithBound only supports nn.Conv2d modules.")
        return output
