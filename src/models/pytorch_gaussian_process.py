"""
Implements a random feature layer to approximate Gaussian processes in PyTorch
"""

import math
import numpy as np
import torch
import torch.nn as nn

class pyRandomFeatureGaussianProcess(nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                num_inducing=1024,
                gp_kernel_scale=2.,  # Length-scale Factor #Paper recommend:2.0
                gp_output_bias=0.,
                normalize_input=True, # Tutorial recommend: True
		        scale_random_features=False,
                gp_cov_momentum=0.999,
                gp_cov_ridge_penalty=0.001,  #Ridge Factor #Paper recommend:0.001
                return_gp_cov=True,
                return_random_features=False,
                ):

        super(pyRandomFeatureGaussianProcess,self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.num_inducing = torch.tensor(num_inducing)
        self.normalize_input = normalize_input
        self.gp_input_scale = 1. / torch.sqrt(torch.tensor(gp_kernel_scale))
        self.gp_feature_scale = torch.sqrt(torch.tensor(2./ float(self.num_inducing)))

        self.return_random_features = return_random_features
        self.return_gp_cov = return_gp_cov

        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty

        self.custom_random_features_activation = torch.cos

        if self.normalize_input:
            normalized_shape = self.in_features
            self._input_norm_layer = nn.LayerNorm(normalized_shape)

        self._random_feature = pyRandomFeatureLayer(
                in_features = self.in_features,
                num_inducing = self.num_inducing,
                activation = self.custom_random_features_activation,
                trainable = False)

        if self.return_gp_cov:
            self._gp_cov_layer = pyLaplaceRandomFeatureCovariance(
                in_num=self.num_inducing,
                momentum=self.gp_cov_momentum,
                ridge_penalty=self.gp_cov_ridge_penalty,
                )

        self._gp_output_layer = nn.Linear(
            in_features=self.num_inducing,
            out_features=self.out_features,
            bias=False)

        self._gp_output_bias = nn.Parameter(
            torch.tensor([self.gp_output_bias] * self.out_features),
            requires_grad=False)

    def reset_covariance_matrix(self):
        self._gp_cov_layer.reset_precision_matrix()

    def forward(self,inputs):
        gp_inputs = inputs

        if self.normalize_input:
            gp_inputs = self._input_norm_layer(gp_inputs)

        gp_input_scale = self.gp_input_scale.to(gp_inputs.dtype)
        #gp_inputs = gp_inputs[0]
        gp_inputs = gp_inputs * gp_input_scale

        gp_feature = self._random_feature(gp_inputs)

        gp_feature_scale = self.gp_feature_scale.to(inputs.dtype)
        if gp_feature_scale:
            gp_feature = gp_feature * gp_feature_scale

        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if self.return_gp_cov:
            gp_covmat = self._gp_cov_layer(gp_feature)

        model_output = [gp_output, ]

        if self.return_gp_cov:
            model_output.append(gp_covmat)
        if self.return_random_features:
            model_output.append(gp_feature)

        return model_output

class pyLaplaceRandomFeatureCovariance(nn.Module):
    def __init__(self,
                 in_num,
                 momentum=0.999,
                 ridge_penalty=1.,
                 ):
        self.ridge_penalty = ridge_penalty
        self.momentum = torch.tensor(momentum)
        super(pyLaplaceRandomFeatureCovariance, self).__init__()

        self.gp_feature_dim = in_num
        self.initial_precision_matrix = self.ridge_penalty * torch.eye(
                                        self.gp_feature_dim, dtype=torch.float32)

        self.precision_matrix = pyAddPrecisionMatrix(gp_feature_dim=self.gp_feature_dim,
                                                     ridge_penalty=self.ridge_penalty)

    def forward(self,x):
        batch_size = x.shape[0]
        if self.training:
            precision_matrix_update_op = self.make_precision_matrix_update_op(
                gp_feature=x,
                precision_matrix=self.precision_matrix
            )
            self.precision_matrix = precision_matrix_update_op
            return torch.eye(batch_size, dtype=torch.float32)
        else:
            return self.compute_predictive_covariance(gp_feature=x)

    def reset_precision_matrix(self):
        self.precision_matrix.back()

    def compute_predictive_covariance(self, gp_feature):
        feature_cov_matrix = torch.inverse(self.precision_matrix())  # Precision_matrix = Phi_tr_T * Phi_tr + S * I
        cov_feature_product = torch.matmul(feature_cov_matrix,  # = S * inverse(t(Phi_tr) * Phi_tr + s * I) X Phi_test_T
                                           gp_feature.T) * self.ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)   # = Phi_test * (S * inverse(t(Phi_tr) * Phi_tr + s * I) X Phi_test_T)
        return gp_cov_matrix

    def make_precision_matrix_update_op(self,
                                      gp_feature,
                                      precision_matrix):
#        gp_feature = gp_feature.to(precision_matrix.device)
        batch_size = gp_feature.shape[0]
        batch_size = torch.tensor(batch_size).to(gp_feature.dtype)
        prob_multiplier = torch.tensor(1.)

        gp_feature_adjusted = torch.sqrt(prob_multiplier) * gp_feature
        precision_matrix_minibatch = torch.matmul(gp_feature_adjusted.T, gp_feature_adjusted)  # Phi_tr_T * Phi_tr
        precision_matrix_data = precision_matrix.forward()
        if self.momentum > 0:
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size 
           #print("device", self.momentum.device, precision_matrix_data.device, precision_matrix_minibatch.device)
            precision_matrix_new = (self.momentum * precision_matrix_data + (1. - self.momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = precision_matrix_data + precision_matrix_minibatch
        precision_matrix.assign(precision_matrix_new)
        return precision_matrix

class pyRandomFeatureLayer(nn.Module):
    def __init__(self,
                 in_features,
                 num_inducing,
                 activation,
                 trainable):
        super(pyRandomFeatureLayer, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=num_inducing,
                                bias=True)
        self.activation = activation
        self.trainable = trainable

        self.initial_weight = torch.tensor(np.random.normal(0, 1, size=(num_inducing, in_features)).astype(np.float64)).float()
        self.initial_bias = torch.tensor(np.random.uniform(0, 2. * math.pi, size=(num_inducing,)).astype(np.float64)).float()
        self.linear.weight.requires_grad = self.trainable
        self.linear.bias.requires_grad = self.trainable

    def forward(self, x):
        self.linear.weight.copy_(self.initial_weight)
        self.linear.bias.copy_(self.initial_bias)
        output = self.linear(x)
        output = self.activation(output)
        return output

class pyAddPrecisionMatrix(nn.Module):
    def __init__(self, gp_feature_dim, ridge_penalty):
        super(pyAddPrecisionMatrix, self).__init__()
        self.gp_feature_dim = gp_feature_dim
        self.ridge_penalty = ridge_penalty
        self.initial_precision_matrix = self.ridge_penalty * torch.eye(self.gp_feature_dim, dtype=torch.float32, device=torch.device("cuda:0"))  # S * I
        self.precision_matrix = nn.Parameter(self.initial_precision_matrix.clone(),
                                             requires_grad=False)

    def back(self):
        self.precision_matrix.data = self.initial_precision_matrix.clone()

    def assign(self,new_precision_matrix):
        self.precision_matrix.data = new_precision_matrix.clone()

    def forward(self):
        return self.precision_matrix.data

def mean_field_logits(logits, covariance_matrix=None, mean_field_factor=1.):
    if mean_field_factor is None or mean_field_factor < 0:
        return logits
    if covariance_matrix is None:
        variances = 1.0
    else:
        variances = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1)
    logits_scale = torch.sqrt(1. + variances * mean_field_factor)

    if len(logits.shape) > 1:
        logits_scale = logits_scale.unsqueeze(-1)

    return logits / logits_scale
