# coding=utf-8
# GAIN with Wasserstein Loss + Gradient Penalty (WGAN-GP) - PyTorch version

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


class Generator(nn.Module):
    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid()
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


class Critic(nn.Module):
    '''Discriminator without Sigmoid — required for WGAN.'''
    def __init__(self, dim, h_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim)
            # Sigmoid 없음 — WGAN 핵심
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, h):
        return self.net(torch.cat([x, h], dim=1))


def compute_gradient_penalty(critic, real, fake, hint, device):
    '''WGAN-GP gradient penalty: (||∇|| - 1)²'''
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    c_interpolated = critic(interpolated, hint)
    gradients = torch.autograd.grad(
        outputs=c_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(c_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def gain_wgan(data_x, gain_parameters):
    '''GAIN with Wasserstein Loss + Gradient Penalty.

    Returns:
        imputed_data : numpy array
        history      : dict with 'c_loss', 'g_loss', 'mse_loss' lists
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_m     = 1 - np.isnan(data_x)
    batch_size = gain_parameters['batch_size']
    hint_rate  = gain_parameters['hint_rate']
    alpha      = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    n_critic   = gain_parameters.get('n_critic', 5)
    lambda_gp  = gain_parameters.get('lambda_gp', 10)

    no, dim = data_x.shape
    h_dim   = dim

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    generator = Generator(dim, h_dim).to(device)
    critic    = Critic(dim, h_dim).to(device)

    C_optimizer = optim.Adam(critic.parameters(),    lr=1e-4, betas=(0.5, 0.9))
    G_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    history = {'c_loss': [], 'g_loss': [], 'mse_loss': []}

    for it in tqdm(range(iterations), desc='WGAN-GP GAIN'):

        # ── Critic update × n_critic ──────────────────────────────
        c_loss_accum = 0.0
        for _ in range(n_critic):
            idx   = sample_batch_index(no, batch_size)
            X_mb  = norm_data_x[idx, :]
            M_mb  = data_m[idx, :]
            Z_mb  = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb  = M_mb * binary_sampler(hint_rate, batch_size, dim)
            X_mb  = M_mb * X_mb + (1 - M_mb) * Z_mb

            Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
            Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
            Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

            critic.train(); generator.eval()
            with torch.no_grad():
                G_s = generator(Xt, Mt)

            real = Xt * Mt + G_s * (1 - Mt)
            fake = G_s

            C_real = critic(real.detach(), Ht)
            C_fake = critic(fake.detach(), Ht)
            W_dist = torch.mean(Mt * C_real) - torch.mean(Mt * C_fake)
            gp     = compute_gradient_penalty(critic, real.detach(), fake.detach(), Ht, device)
            C_loss = -W_dist + lambda_gp * gp

            C_optimizer.zero_grad()
            C_loss.backward()
            C_optimizer.step()
            c_loss_accum += C_loss.item()

        # ── Generator update × 1 ──────────────────────────────────
        idx   = sample_batch_index(no, batch_size)
        X_mb  = norm_data_x[idx, :]
        M_mb  = data_m[idx, :]
        Z_mb  = uniform_sampler(0, 0.01, batch_size, dim)
        H_mb  = M_mb * binary_sampler(hint_rate, batch_size, dim)
        X_mb  = M_mb * X_mb + (1 - M_mb) * Z_mb

        Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
        Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

        generator.train(); critic.eval()
        G_s     = generator(Xt, Mt)
        Hat_X   = Xt * Mt + G_s * (1 - Mt)
        C_score = critic(Hat_X, Ht)

        G_loss_adv = -torch.mean((1 - Mt) * C_score)
        MSE_loss   = torch.mean((Mt * Xt - Mt * G_s) ** 2) / torch.mean(Mt)
        G_loss     = G_loss_adv + alpha * MSE_loss

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # 100 iteration마다 loss 기록
        if (it + 1) % 100 == 0:
            history['c_loss'].append(c_loss_accum / n_critic)
            history['g_loss'].append(G_loss.item())
            history['mse_loss'].append(MSE_loss.item())

    # ── Inference ─────────────────────────────────────────────────
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