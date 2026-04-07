# coding=utf-8
# Original GAIN (PyTorch) — loss history 기록 버전
# gain.py 원본을 건드리지 않고 history 반환만 추가한 래퍼

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


class Generator(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),   nn.ReLU(),
            nn.Linear(h_dim, dim),     nn.Sigmoid()
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


class Discriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),   nn.ReLU(),
            nn.Linear(h_dim, dim),     nn.Sigmoid()
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, x, h):
        return self.net(torch.cat([x, h], dim=1))


def gain_with_history(data_x, gain_parameters):
    '''Original GAIN — history 반환 추가 버전.
    gain.py 원본은 그대로 유지하고 이 파일에서 history를 기록.

    Returns:
        imputed_data, history (dict: d_loss, g_loss, mse_loss)
    '''
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_m  = 1 - np.isnan(data_x)

    batch_size = gain_parameters['batch_size']
    hint_rate  = gain_parameters['hint_rate']
    alpha      = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    no, dim = data_x.shape
    h_dim   = dim

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    generator     = Generator(dim, h_dim).to(device)
    discriminator = Discriminator(dim, h_dim).to(device)

    D_optimizer = optim.Adam(discriminator.parameters())
    G_optimizer = optim.Adam(generator.parameters())

    history = {'d_loss': [], 'g_loss': [], 'mse_loss': []}

    for it in tqdm(range(iterations), desc='Original GAIN'):
        idx   = sample_batch_index(no, batch_size)
        X_mb  = norm_data_x[idx, :]
        M_mb  = data_m[idx, :]
        Z_mb  = uniform_sampler(0, 0.01, batch_size, dim)
        H_mb  = M_mb * binary_sampler(hint_rate, batch_size, dim)
        X_mb  = M_mb * X_mb + (1 - M_mb) * Z_mb

        Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
        Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

        # Discriminator step
        discriminator.train(); generator.eval()
        with torch.no_grad():
            G_s = generator(Xt, Mt)
        Hat_X  = Xt * Mt + G_s * (1 - Mt)
        D_prob = discriminator(Hat_X.detach(), Ht)
        D_loss = -torch.mean(Mt * torch.log(D_prob + 1e-8) +
                             (1 - Mt) * torch.log(1 - D_prob + 1e-8))
        D_optimizer.zero_grad(); D_loss.backward(); D_optimizer.step()

        # Generator step
        generator.train(); discriminator.eval()
        G_s    = generator(Xt, Mt)
        Hat_X  = Xt * Mt + G_s * (1 - Mt)
        D_prob = discriminator(Hat_X, Ht)
        G_loss_adv = -torch.mean((1 - Mt) * torch.log(D_prob + 1e-8))
        MSE_loss   = torch.mean((Mt * Xt - Mt * G_s) ** 2) / torch.mean(Mt)
        G_loss     = G_loss_adv + alpha * MSE_loss
        G_optimizer.zero_grad(); G_loss.backward(); G_optimizer.step()

        if (it + 1) % 100 == 0:
            history['d_loss'].append(D_loss.item())
            history['g_loss'].append(G_loss.item())
            history['mse_loss'].append(MSE_loss.item())

    # Inference
    generator.eval()
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    X_mb = data_m * norm_data_x + (1 - data_m) * Z_mb
    Xt   = torch.tensor(X_mb, dtype=torch.float32).to(device)
    Mt   = torch.tensor(data_m, dtype=torch.float32).to(device)
    with torch.no_grad():
        imputed_data = generator(Xt, Mt).cpu().numpy()

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data, history
