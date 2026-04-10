# coding=utf-8
# gain_kd.py
# ─────────────────────────────────────────────────────────────────────
# 이 파일 하나로 아래 역할을 모두 담당합니다:
#   1. gain_original_with_history.py (삭제됨) 의 역할
#      → train_teacher_original(..., return_history=True) 로 대체
#   2. Teacher 학습: Original GAIN / WGAN-GP GAIN
#   3. Student 학습: Knowledge Distillation (Original / WGAN-GP)
#   4. 공통 Inference
#
# [gain.py 와의 관계]
#   gain.py 의 gain() 함수는 외부(main)에서 "원본 코드 직접 호출 확인"
#   용도로만 사용됩니다. 실제 Teacher 학습은 이 파일의
#   train_teacher_original() 을 사용합니다.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


# ══════════════════════════════════════════════════════════════════════
# 공통 네트워크 (h_dim으로 Teacher/Student 크기 분기)
# ══════════════════════════════════════════════════════════════════════

class GAINGenerator(nn.Module):
    '''Teacher/Student 공용 Generator.'''
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),   nn.ReLU(),
            nn.Linear(h_dim, dim),     nn.Sigmoid()
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


class GAINDiscriminator(nn.Module):
    '''Original GAIN용 Discriminator (Sigmoid 포함).'''
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),   nn.ReLU(),
            nn.Linear(h_dim, dim),     nn.Sigmoid()
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, h):
        return self.net(torch.cat([x, h], dim=1))


class WGANCritic(nn.Module):
    '''WGAN-GP용 Critic (Sigmoid 없음).'''
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),   nn.ReLU(),
            nn.Linear(h_dim, dim)      # Sigmoid 없음 — WGAN 핵심
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, h):
        return self.net(torch.cat([x, h], dim=1))


def count_parameters(model):
    '''파라미터 수 반환 (frozen 여부 무관).'''
    return sum(p.numel() for p in model.parameters())


# ══════════════════════════════════════════════════════════════════════
# Gradient Penalty (WGAN-GP)
# ══════════════════════════════════════════════════════════════════════

def compute_gradient_penalty(critic, real, fake, hint, device):
    '''WGAN-GP gradient penalty: (||∇|| - 1)²'''
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    c_out = critic(interpolated, hint)
    grads = torch.autograd.grad(
        outputs=c_out, inputs=interpolated,
        grad_outputs=torch.ones_like(c_out),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return ((grads.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()


# ══════════════════════════════════════════════════════════════════════
# Teacher: Original GAIN
# (구 gain_original_with_history.py 의 gain_with_history() 역할 통합)
# ══════════════════════════════════════════════════════════════════════

def train_teacher_original(data_x, params, device, return_history=False):
    '''Original GAIN Teacher 학습. h_dim = dim (full size).

    Args:
        return_history: True이면 (G, norm_data_x, data_m, norm_params, history)
                        False이면 (G, norm_data_x, data_m, norm_params)
                        → gain_original_with_history.py 를 대체
    '''
    data_m  = 1 - np.isnan(data_x)
    no, dim = data_x.shape
    h_dim   = dim  # Teacher: full size

    norm_data, norm_params = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    G = GAINGenerator(dim, h_dim).to(device)
    D = GAINDiscriminator(dim, h_dim).to(device)
    G_opt = optim.Adam(G.parameters())
    D_opt = optim.Adam(D.parameters())

    history = {'d_loss': [], 'g_loss': [], 'mse_loss': []}

    for it in tqdm(range(params['iterations']), desc='[Teacher] Original GAIN'):
        idx  = sample_batch_index(no, params['batch_size'])
        X_mb = norm_data_x[idx, :]
        M_mb = data_m[idx, :]
        Z_mb = uniform_sampler(0, 0.01, params['batch_size'], dim)
        H_mb = M_mb * binary_sampler(params['hint_rate'], params['batch_size'], dim)
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
        Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

        # D step
        D.train(); G.eval()
        with torch.no_grad(): Gs = G(Xt, Mt)
        Hat_X  = Xt * Mt + Gs * (1 - Mt)
        D_prob = D(Hat_X.detach(), Ht)
        D_loss = -torch.mean(Mt * torch.log(D_prob + 1e-8) +
                             (1 - Mt) * torch.log(1 - D_prob + 1e-8))
        D_opt.zero_grad(); D_loss.backward(); D_opt.step()

        # G step
        G.train(); D.eval()
        Gs     = G(Xt, Mt)
        Hat_X  = Xt * Mt + Gs * (1 - Mt)
        D_prob = D(Hat_X, Ht)
        MSE    = torch.mean((Mt * Xt - Mt * Gs) ** 2) / torch.mean(Mt)
        G_loss = -torch.mean((1 - Mt) * torch.log(D_prob + 1e-8)) + params['alpha'] * MSE
        G_opt.zero_grad(); G_loss.backward(); G_opt.step()

        if (it + 1) % 100 == 0:
            history['d_loss'].append(D_loss.item())
            history['g_loss'].append(G_loss.item())
            history['mse_loss'].append(MSE.item())

    G.eval()
    print(f'  Teacher (Original GAIN) params: {count_parameters(G):,}')

    if return_history:
        return G, norm_data_x, data_m, norm_params, history
    return G, norm_data_x, data_m, norm_params


# ══════════════════════════════════════════════════════════════════════
# Teacher: WGAN-GP GAIN
# ══════════════════════════════════════════════════════════════════════

def train_teacher_wgan(data_x, params, device, return_history=False):
    '''WGAN-GP GAIN Teacher 학습. h_dim = dim (full size).'''
    data_m  = 1 - np.isnan(data_x)
    no, dim = data_x.shape
    h_dim   = dim

    norm_data, norm_params = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    G = GAINGenerator(dim, h_dim).to(device)
    C = WGANCritic(dim, h_dim).to(device)
    G_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    C_opt = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    n_critic  = params.get('n_critic', 5)
    lambda_gp = params.get('lambda_gp', 10)
    history   = {'c_loss': [], 'g_loss': [], 'mse_loss': []}

    for it in tqdm(range(params['iterations']), desc='[Teacher] WGAN-GP GAIN'):
        c_loss_accum = 0.0
        for _ in range(n_critic):
            idx  = sample_batch_index(no, params['batch_size'])
            X_mb = norm_data_x[idx, :]
            M_mb = data_m[idx, :]
            Z_mb = uniform_sampler(0, 0.01, params['batch_size'], dim)
            H_mb = M_mb * binary_sampler(params['hint_rate'], params['batch_size'], dim)
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
            Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
            Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

            C.train(); G.eval()
            with torch.no_grad(): Gs = G(Xt, Mt)
            real   = Xt * Mt + Gs * (1 - Mt)
            fake   = Gs
            W_dist = torch.mean(Mt * C(real.detach(), Ht)) - \
                     torch.mean(Mt * C(fake.detach(), Ht))
            gp     = compute_gradient_penalty(C, real.detach(), fake.detach(), Ht, device)
            C_loss = -W_dist + lambda_gp * gp
            C_opt.zero_grad(); C_loss.backward(); C_opt.step()
            c_loss_accum += C_loss.item()

        idx  = sample_batch_index(no, params['batch_size'])
        X_mb = norm_data_x[idx, :]
        M_mb = data_m[idx, :]
        Z_mb = uniform_sampler(0, 0.01, params['batch_size'], dim)
        H_mb = M_mb * binary_sampler(params['hint_rate'], params['batch_size'], dim)
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
        Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

        G.train(); C.eval()
        Gs    = G(Xt, Mt)
        Hat_X = Xt * Mt + Gs * (1 - Mt)
        MSE   = torch.mean((Mt * Xt - Mt * Gs) ** 2) / torch.mean(Mt)
        G_loss = -torch.mean((1 - Mt) * C(Hat_X, Ht)) + params['alpha'] * MSE
        G_opt.zero_grad(); G_loss.backward(); G_opt.step()

        if (it + 1) % 100 == 0:
            history['c_loss'].append(c_loss_accum / n_critic)
            history['g_loss'].append(G_loss.item())
            history['mse_loss'].append(MSE.item())

    G.eval()
    print(f'  Teacher (WGAN-GP GAIN) params: {count_parameters(G):,}')

    if return_history:
        return G, norm_data_x, data_m, norm_params, history
    return G, norm_data_x, data_m, norm_params


# ══════════════════════════════════════════════════════════════════════
# Student + Knowledge Distillation
# ══════════════════════════════════════════════════════════════════════

def train_student_kd(data_x, teacher_G, params, use_wgan, device,
                     temperature=2.0, kd_weight=0.5):
    '''KD로 Student Generator 학습. Student h_dim = dim // 2 (0.5배).

    Student Loss = kd_weight     * KD Loss   (Teacher soft target 모방)
                 + (1-kd_weight) * Task Loss  (관측값 복원 MSE)
                 + alpha         * Adv Loss   (GAN 적대 학습)
    '''
    data_m  = 1 - np.isnan(data_x)
    no, dim = data_x.shape
    h_dim_s = max(dim // 2, 8)  # Student: 0.5배 (최소 8)

    norm_data, norm_params = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    S_G = GAINGenerator(dim, h_dim_s).to(device)
    if use_wgan:
        S_D     = WGANCritic(dim, h_dim_s).to(device)
        S_G_opt = optim.Adam(S_G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        S_D_opt = optim.Adam(S_D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    else:
        S_D     = GAINDiscriminator(dim, h_dim_s).to(device)
        S_G_opt = optim.Adam(S_G.parameters())
        S_D_opt = optim.Adam(S_D.parameters())

    # Teacher frozen
    teacher_G.eval()
    for p in teacher_G.parameters():
        p.requires_grad = False

    n_critic  = params.get('n_critic', 5)
    lambda_gp = params.get('lambda_gp', 10)
    desc      = '[Student-KD] WGAN-GP' if use_wgan else '[Student-KD] Original'
    kd_losses, task_losses = [], []

    for it in tqdm(range(params['iterations']), desc=desc):

        d_steps = n_critic if use_wgan else 1

        # ── D/C 업데이트 ──────────────────────────────────────────
        for _ in range(d_steps):
            idx  = sample_batch_index(no, params['batch_size'])
            X_mb = norm_data_x[idx, :]
            M_mb = data_m[idx, :]
            Z_mb = uniform_sampler(0, 0.01, params['batch_size'], dim)
            H_mb = M_mb * binary_sampler(params['hint_rate'], params['batch_size'], dim)
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
            Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
            Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

            S_D.train(); S_G.eval()
            with torch.no_grad(): S_Gs = S_G(Xt, Mt)
            Hat_X = Xt * Mt + S_Gs * (1 - Mt)

            if use_wgan:
                W_dist = torch.mean(Mt * S_D(Hat_X.detach(), Ht)) - \
                         torch.mean(Mt * S_D(S_Gs.detach(), Ht))
                gp     = compute_gradient_penalty(S_D, Hat_X.detach(), S_Gs.detach(), Ht, device)
                D_loss = -W_dist + lambda_gp * gp
            else:
                D_prob = S_D(Hat_X.detach(), Ht)
                D_loss = -torch.mean(Mt * torch.log(D_prob + 1e-8) +
                                     (1 - Mt) * torch.log(1 - D_prob + 1e-8))
            S_D_opt.zero_grad(); D_loss.backward(); S_D_opt.step()

        # ── G 업데이트 (KD Loss 포함) ─────────────────────────────
        idx  = sample_batch_index(no, params['batch_size'])
        X_mb = norm_data_x[idx, :]
        M_mb = data_m[idx, :]
        Z_mb = uniform_sampler(0, 0.01, params['batch_size'], dim)
        H_mb = M_mb * binary_sampler(params['hint_rate'], params['batch_size'], dim)
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        Xt = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt = torch.tensor(M_mb, dtype=torch.float32).to(device)
        Ht = torch.tensor(H_mb, dtype=torch.float32).to(device)

        S_G.train(); S_D.eval()
        with torch.no_grad(): T_out = teacher_G(Xt, Mt)
        S_out = S_G(Xt, Mt)
        Hat_X = Xt * Mt + S_out * (1 - Mt)

        # KD Loss: missing 위치에서 Teacher soft target 모방
        kd_loss  = torch.mean((1 - Mt) * (S_out / temperature - T_out / temperature) ** 2)
        # Task Loss: 관측 위치 복원 MSE
        mse_task = torch.mean((Mt * Xt - Mt * S_out) ** 2) / torch.mean(Mt)
        # Adversarial Loss
        if use_wgan:
            adv_loss = -torch.mean((1 - Mt) * S_D(Hat_X, Ht))
        else:
            D_prob   = S_D(Hat_X, Ht)
            adv_loss = -torch.mean((1 - Mt) * torch.log(D_prob + 1e-8))

        G_loss = kd_weight * kd_loss + (1 - kd_weight) * mse_task + params['alpha'] * adv_loss
        S_G_opt.zero_grad(); G_loss.backward(); S_G_opt.step()

        if (it + 1) % 100 == 0:
            kd_losses.append(kd_loss.item())
            task_losses.append(mse_task.item())

    S_G.eval()
    mode     = 'WGAN-GP' if use_wgan else 'Original'
    s_params = count_parameters(S_G)
    t_params = count_parameters(teacher_G)
    ratio    = (s_params / t_params * 100) if t_params > 0 else 0.0
    print(f'  Student ({mode}) params: {s_params:,}  (Teacher 대비 {ratio:.1f}%)')

    return S_G, norm_data_x, data_m, norm_params, kd_losses, task_losses


# ══════════════════════════════════════════════════════════════════════
# 공통 Inference
# ══════════════════════════════════════════════════════════════════════

def inference(G, norm_data_x, data_m, norm_params, data_x, device):
    '''학습된 Generator로 imputation 수행.'''
    G.eval()
    no, dim = norm_data_x.shape
    Z_mb    = uniform_sampler(0, 0.01, no, dim)
    X_mb    = data_m * norm_data_x + (1 - data_m) * Z_mb
    Xt      = torch.tensor(X_mb, dtype=torch.float32).to(device)
    Mt      = torch.tensor(data_m, dtype=torch.float32).to(device)
    with torch.no_grad():
        imputed = G(Xt, Mt).cpu().numpy()
    imputed = data_m * norm_data_x + (1 - data_m) * imputed
    imputed = renormalization(imputed, norm_params)
    imputed = rounding(imputed, data_x)
    return imputed
