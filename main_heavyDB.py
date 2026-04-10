# coding=utf-8
# main_heavyDB.py
# 대용량 데이터셋(HIGGS, Criteo) 전용 실행 파일

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader_heavy import data_loader_heavy
from gain_kd import (train_teacher_original, train_teacher_wgan,
                     train_student_kd, inference, count_parameters)
from utils import normalization, rmse_loss


def mae_loss(ori_data, imputed_data, data_m):
    ori_norm, norm_params = normalization(ori_data)
    imp_norm, _           = normalization(imputed_data, norm_params)
    mask  = (1 - data_m)
    denom = np.sum(mask)
    return np.sum(np.abs(mask * ori_norm - mask * imp_norm)) / denom if denom > 0 else 0.0


def print_results(results):
    print('\n' + '=' * 65)
    print(f"  {'Model':30s} {'RMSE':>8s} {'MAE':>8s} {'Params':>10s}")
    print('-' * 65)
    for name, m in results.items():
        params_str = f"{m['params']:,}" if m.get('params', 0) > 0 else 'N/A'
        print(f"  {name:30s} {m['rmse']:>8.4f} {m['mae']:>8.4f} {params_str:>10s}")
    print('=' * 65)

    # KD 효율성 요약
    pairs = [
        ('Original GAIN (Teacher)', 'Student Original + KD'),
        ('WGAN-GP (Teacher)',        'Student WGAN-GP + KD'),
    ]
    print('\n[KD 효율성 분석]')
    for t_key, s_key in pairs:
        if t_key in results and s_key in results:
            rmse_gap  = results[s_key]['rmse'] - results[t_key]['rmse']
            t_p = results[t_key].get('params', 0)
            s_p = results[s_key].get('params', 0)
            param_red = (t_p - s_p) / t_p * 100 if t_p > 0 else 0
            print(f"  {t_key} → {s_key}")
            print(f"    RMSE 격차: {rmse_gap:+.4f}  파라미터 감소: {param_red:.1f}%")


def plot_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names  = list(results.keys())
    rmses  = [results[n]['rmse'] for n in names]
    maes   = [results[n]['mae']  for n in names]
    colors = ['steelblue', 'mediumseagreen', 'tomato', 'darkorange']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Heavy DB — Teacher vs Student Comparison', fontsize=13, fontweight='bold')

    for ax, vals, title, ylabel in zip(
        axes, [rmses, maes], ['RMSE Comparison', 'MAE Comparison'], ['RMSE ↓', 'MAE ↓']
    ):
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)], width=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n[그래프 저장] → {save_path}')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{"="*60}')
    print(f'  LightGAIN Heavy DB: {args.data_name.upper()}  |  device: {device}')
    print(f'{"="*60}')

    ori_data_x, miss_data_x, data_m = data_loader_heavy(
        args.data_name, args.miss_rate, max_samples=args.max_samples
    )

    params = {
        'batch_size': args.batch_size,
        'iterations': args.iterations,
        'alpha'     : args.alpha,
        'hint_rate' : args.hint_rate,
        'n_critic'  : args.n_critic,
        'lambda_gp' : args.lambda_gp,
    }

    results = {}

    # ── [1] Original GAIN Teacher ─────────────────────────────────
    print('\n[1/4] Original GAIN Teacher 학습...')
    T_orig, nd_orig, dm_orig, np_orig = train_teacher_original(
        miss_data_x, params, device)
    imp = inference(T_orig, nd_orig, dm_orig, np_orig, miss_data_x, device)
    results['Original GAIN (Teacher)'] = {
        'rmse'  : rmse_loss(ori_data_x, imp, data_m),
        'mae'   : mae_loss(ori_data_x, imp, data_m),
        'params': count_parameters(T_orig),
    }

    # ── [2] Student Original + KD ─────────────────────────────────
    print('\n[2/4] Student Original GAIN + KD 학습 (0.5x)...')
    S_orig, nd_s, dm_s, np_s, _, _ = train_student_kd(
        miss_data_x, T_orig, params, use_wgan=False, device=device,
        temperature=args.temperature, kd_weight=args.kd_weight)
    imp = inference(S_orig, nd_s, dm_s, np_s, miss_data_x, device)
    results['Student Original + KD'] = {
        'rmse'  : rmse_loss(ori_data_x, imp, data_m),
        'mae'   : mae_loss(ori_data_x, imp, data_m),
        'params': count_parameters(S_orig),
    }

    # ── [3] WGAN-GP Teacher ───────────────────────────────────────
    print('\n[3/4] WGAN-GP GAIN Teacher 학습...')
    T_wgan, nd_wgan, dm_wgan, np_wgan = train_teacher_wgan(
        miss_data_x, params, device)
    imp = inference(T_wgan, nd_wgan, dm_wgan, np_wgan, miss_data_x, device)
    results['WGAN-GP (Teacher)'] = {
        'rmse'  : rmse_loss(ori_data_x, imp, data_m),
        'mae'   : mae_loss(ori_data_x, imp, data_m),
        'params': count_parameters(T_wgan),
    }

    # ── [4] Student WGAN-GP + KD ──────────────────────────────────
    print('\n[4/4] Student WGAN-GP GAIN + KD 학습 (0.5x)...')
    S_wgan, nd_sw, dm_sw, np_sw, _, _ = train_student_kd(
        miss_data_x, T_wgan, params, use_wgan=True, device=device,
        temperature=args.temperature, kd_weight=args.kd_weight)
    imp = inference(S_wgan, nd_sw, dm_sw, np_sw, miss_data_x, device)
    results['Student WGAN-GP + KD'] = {
        'rmse'  : rmse_loss(ori_data_x, imp, data_m),
        'mae'   : mae_loss(ori_data_x, imp, data_m),
        'params': count_parameters(S_wgan),
    }

    print_results(results)
    plot_results(results, save_path=f'images/results_heavy_{args.data_name}_missrate80.png')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',   choices=['higgs', 'criteo'], default='higgs')
    parser.add_argument('--miss_rate',   default=0.8,   type=float)
    parser.add_argument('--max_samples', default=50000, type=int)
    parser.add_argument('--batch_size',  default=256,   type=int)
    parser.add_argument('--iterations',  default=5000,  type=int)
    parser.add_argument('--hint_rate',   default=0.9,   type=float)
    parser.add_argument('--alpha',       default=10,    type=float)
    parser.add_argument('--n_critic',    default=5,     type=int)
    parser.add_argument('--lambda_gp',   default=10,    type=float)
    parser.add_argument('--temperature', default=2.0,   type=float)
    parser.add_argument('--kd_weight',   default=0.5,   type=float)
    args = parser.parse_args()
    main(args)
