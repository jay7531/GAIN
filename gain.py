# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''GAIN function (PyTorch version).
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com

PyTorch conversion: Original TF1 code converted to PyTorch.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


# ─────────────────────────────────────────────
# Generator Network
# ─────────────────────────────────────────────
class Generator(nn.Module):
    '''Generator: takes (X, M) as input and outputs imputed values.

    Input:  [data; mask]  → dim * 2
    Output: imputed data  → dim
    '''
    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),   # Input: data + mask
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid()                 # MinMax normalized output
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)   # Concatenate data and mask
        return self.net(inputs)


# ─────────────────────────────────────────────
# Discriminator Network
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    '''Discriminator: takes (X_hat, H) as input and predicts which
    components were observed vs. imputed.

    Input:  [data_hat; hint]  → dim * 2
    Output: probability       → dim
    '''
    def __init__(self, dim, h_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),   # Input: filled data + hint
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)   # Concatenate data and hint
        return self.net(inputs)


# ─────────────────────────────────────────────
# GAIN main function
# ─────────────────────────────────────────────
def gain(data_x, gain_parameters):
    '''Impute missing values in data_x using GAIN (PyTorch).

    Args:
        - data_x: original data with missing values  (numpy array)
        - gain_parameters: dict with keys:
            - batch_size : int
            - hint_rate  : float
            - alpha      : float  (reconstruction loss weight)
            - iterations : int

    Returns:
        - imputed_data: imputed data  (numpy array)
    '''
    # ── Device setup ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Mask matrix  (1 = observed, 0 = missing) ──────────────────
    data_m = 1 - np.isnan(data_x)

    # ── Hyperparameters ───────────────────────────────────────────
    batch_size = gain_parameters['batch_size']
    hint_rate  = gain_parameters['hint_rate']
    alpha      = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    no, dim = data_x.shape
    h_dim   = dim   # Hidden layer size = feature dimension (same as original)

    # ── Normalization ─────────────────────────────────────────────
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # ── Model initialization ──────────────────────────────────────
    generator     = Generator(dim, h_dim).to(device)
    discriminator = Discriminator(dim, h_dim).to(device)

    # ── Optimizers (Adam, same as original) ───────────────────────
    D_optimizer = optim.Adam(discriminator.parameters())
    G_optimizer = optim.Adam(generator.parameters())

    # ── Training loop ─────────────────────────────────────────────
    for it in tqdm(range(iterations)):

        # Sample mini-batch indices
        batch_idx = sample_batch_index(no, batch_size)

        X_mb = norm_data_x[batch_idx, :]   # (batch, dim)
        M_mb = data_m[batch_idx, :]        # (batch, dim)

        # Random noise for missing positions
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

        # Hint vector: only reveal a subset of the mask to discriminator
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp             # (batch, dim)

        # Replace missing entries with noise
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb   # (batch, dim)

        # Convert to tensors
        X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)
        M_mb = torch.tensor(M_mb, dtype=torch.float32).to(device)
        H_mb = torch.tensor(H_mb, dtype=torch.float32).to(device)

        # ── Discriminator step ────────────────────────────────────
        discriminator.train()
        generator.eval()

        with torch.no_grad():
            G_sample = generator(X_mb, M_mb)

        # Combine observed data with generated imputations
        Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)

        D_prob = discriminator(Hat_X.detach(), H_mb)

        # Discriminator loss: cross-entropy on observed vs. missing positions
        D_loss = -torch.mean(
            M_mb * torch.log(D_prob + 1e-8) +
            (1 - M_mb) * torch.log(1.0 - D_prob + 1e-8)
        )

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # ── Generator step ────────────────────────────────────────
        generator.train()
        discriminator.eval()

        G_sample = generator(X_mb, M_mb)
        Hat_X    = X_mb * M_mb + G_sample * (1 - M_mb)
        D_prob   = discriminator(Hat_X, H_mb)

        # Generator adversarial loss (fool the discriminator on missing entries)
        G_loss_adv = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))

        # Reconstruction loss on observed entries
        MSE_loss = torch.mean((M_mb * X_mb - M_mb * G_sample) ** 2) / torch.mean(M_mb)

        G_loss = G_loss_adv + alpha * MSE_loss

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    # ── Imputation (inference) ────────────────────────────────────
    generator.eval()

    Z_mb      = uniform_sampler(0, 0.01, no, dim)
    M_mb      = data_m
    X_mb      = norm_data_x
    X_mb      = M_mb * X_mb + (1 - M_mb) * Z_mb

    X_mb_t = torch.tensor(X_mb, dtype=torch.float32).to(device)
    M_mb_t = torch.tensor(M_mb, dtype=torch.float32).to(device)

    with torch.no_grad():
        imputed_data = generator(X_mb_t, M_mb_t).cpu().numpy()

    # Keep observed values; fill only missing positions
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalize and round
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data