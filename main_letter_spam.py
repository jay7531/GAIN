# coding=utf-8
# Main — Original GAIN vs WGAN-GP GAIN 성능 비교

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import data_loader
from gain import gain                                      # 원본 gain.py — 그대로 유지
from gain_original_with_history import gain_with_history   # history 기록용 래퍼
from gain_wgan import gain_wgan                            # WGAN-GP 버전
from utils import normalization


# ── 추가 지표: MAE ────────────────────────────────────────────────────
def mae_loss(ori_data, imputed_data, data_m):
    '''결측 위치만 골라서 MAE 계산.'''
    ori_norm, norm_params   = normalization(ori_data)
    imp_norm, _             = normalization(imputed_data, norm_params)
    missing_mask            = (1 - data_m)
    nominator               = np.sum(np.abs(missing_mask * ori_norm -
                                            missing_mask * imp_norm))
    denominator             = np.sum(missing_mask)
    return nominator / float(denominator)


# ── 성능 비교 테이블 출력 ──────────────────────────────────────────────
def print_comparison_table(results):
    print('\n' + '=' * 55)
    print(f"{'':20s} {'RMSE':>10s} {'MAE':>10s} {'Improvement':>12s}")
    print('-' * 55)

    base_rmse = results['Original GAIN']['rmse']
    base_mae  = results['Original GAIN']['mae']

    for name, metrics in results.items():
        rmse = metrics['rmse']
        mae  = metrics['mae']
        if name == 'Original GAIN':
            imp_str = '  (baseline)'
        else:
            rmse_imp = (base_rmse - rmse) / base_rmse * 100
            mae_imp  = (base_mae  - mae)  / base_mae  * 100
            imp_str  = f'RMSE {rmse_imp:+.1f}%'
        print(f"{name:20s} {rmse:>10.4f} {mae:>10.4f} {imp_str:>12s}")
    print('=' * 55 + '\n')


# ── Loss 곡선 + 성능 비교 그래프 ──────────────────────────────────────
def plot_results(orig_history, wgan_history, results, save_path='results.png'):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Original GAIN vs WGAN-GP GAIN', fontsize=15, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    iters = range(1, len(orig_history['d_loss']) + 1)

    # ── 1. Discriminator / Critic Loss ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, orig_history['d_loss'], label='Original D Loss', color='steelblue')
    ax1.plot(iters, wgan_history['c_loss'], label='WGAN Critic Loss', color='tomato')
    ax1.set_title('Discriminator / Critic Loss')
    ax1.set_xlabel('Iteration (×100)')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Generator Loss ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, orig_history['g_loss'], label='Original G Loss', color='steelblue')
    ax2.plot(iters, wgan_history['g_loss'], label='WGAN G Loss',     color='tomato')
    ax2.set_title('Generator Loss')
    ax2.set_xlabel('Iteration (×100)')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── 3. MSE Loss ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iters, orig_history['mse_loss'], label='Original MSE', color='steelblue')
    ax3.plot(iters, wgan_history['mse_loss'], label='WGAN MSE',     color='tomato')
    ax3.set_title('Reconstruction MSE Loss')
    ax3.set_xlabel('Iteration (×100)')
    ax3.set_ylabel('MSE')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. RMSE 비교 바 차트 ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    names  = list(results.keys())
    rmses  = [results[n]['rmse'] for n in names]
    colors = ['steelblue', 'tomato']
    bars   = ax4.bar(names, rmses, color=colors, width=0.4)
    ax4.set_title('RMSE Comparison')
    ax4.set_ylabel('RMSE (lower is better)')
    for bar, val in zip(bars, rmses):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, max(rmses) * 1.25)
    ax4.grid(True, alpha=0.3, axis='y')

    # ── 5. MAE 비교 바 차트 ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    maes   = [results[n]['mae'] for n in names]
    bars   = ax5.bar(names, maes, color=colors, width=0.4)
    ax5.set_title('MAE Comparison')
    ax5.set_ylabel('MAE (lower is better)')
    for bar, val in zip(bars, maes):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.set_ylim(0, max(maes) * 1.25)
    ax5.grid(True, alpha=0.3, axis='y')

    # ── 6. Improvement 요약 텍스트 ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    orig_rmse = results['Original GAIN']['rmse']
    orig_mae  = results['Original GAIN']['mae']
    wgan_rmse = results['WGAN-GP GAIN']['rmse']
    wgan_mae  = results['WGAN-GP GAIN']['mae']
    rmse_imp  = (orig_rmse - wgan_rmse) / orig_rmse * 100
    mae_imp   = (orig_mae  - wgan_mae)  / orig_mae  * 100
    winner    = 'WGAN-GP GAIN' if wgan_rmse < orig_rmse else 'Original GAIN'

    summary = (
        f"  ── Summary ──\n\n"
        f"  RMSE\n"
        f"  Original : {orig_rmse:.4f}\n"
        f"  WGAN-GP  : {wgan_rmse:.4f}\n"
        f"  Change   : {rmse_imp:+.2f}%\n\n"
        f"  MAE\n"
        f"  Original : {orig_mae:.4f}\n"
        f"  WGAN-GP  : {wgan_mae:.4f}\n"
        f"  Change   : {mae_imp:+.2f}%\n\n"
        f"  Best Model: {winner}"
    )
    ax6.text(0.05, 0.95, summary,
             transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n[결과 그래프 저장됨] → {save_path}')


# ── Main ──────────────────────────────────────────────────────────────
def main(args):
    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {
        'batch_size': args.batch_size,
        'hint_rate' : args.hint_rate,
        'alpha'     : args.alpha,
        'iterations': args.iterations,
        'n_critic'  : args.n_critic,
        'lambda_gp' : args.lambda_gp,
    }

    # 데이터 로드 (두 모델이 동일한 데이터로 비교)
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

    # ── 1. Original GAIN (gain.py 원본 — 그대로 사용) ──────────────
    print('\n[1/2] Original GAIN 학습 시작...')
    orig_imputed, orig_history = gain_with_history(miss_data_x, gain_parameters)

    # 원본 gain() 결과도 함께 확인 (gain.py 직접 호출)
    orig_rmse_check = gain(miss_data_x, gain_parameters)

    from utils import rmse_loss
    orig_rmse = rmse_loss(ori_data_x, orig_imputed, data_m)
    orig_mae  = mae_loss(ori_data_x, orig_imputed, data_m)

    print(f'\n  ▶ Original GAIN  RMSE: {orig_rmse:.4f}  MAE: {orig_mae:.4f}')

    # ── 2. WGAN-GP GAIN ────────────────────────────────────────────
    print('\n[2/2] WGAN-GP GAIN 학습 시작...')
    wgan_imputed, wgan_history = gain_wgan(miss_data_x, gain_parameters)

    wgan_rmse = rmse_loss(ori_data_x, wgan_imputed, data_m)
    wgan_mae  = mae_loss(ori_data_x, wgan_imputed, data_m)

    print(f'\n  ▶ WGAN-GP GAIN   RMSE: {wgan_rmse:.4f}  MAE: {wgan_mae:.4f}')

    # ── 3. 성능 비교 테이블 출력 ───────────────────────────────────
    results = {
        'Original GAIN': {'rmse': orig_rmse, 'mae': orig_mae},
        'WGAN-GP GAIN' : {'rmse': wgan_rmse, 'mae': wgan_mae},
    }
    print_comparison_table(results)

    # ── 4. 그래프 저장 ─────────────────────────────────────────────
    plot_results(orig_history, wgan_history, results,
                 save_path=f'results_{data_name}.png')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',  choices=['letter', 'spam'], default='spam', type=str)
    parser.add_argument('--miss_rate',  default=0.2,   type=float)
    parser.add_argument('--batch_size', default=128,   type=int)
    parser.add_argument('--hint_rate',  default=0.9,   type=float)
    parser.add_argument('--alpha',      default=100,   type=float)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--n_critic',   default=5,     type=int,
                        help='critic updates per generator update (WGAN)')
    parser.add_argument('--lambda_gp',  default=10,    type=float,
                        help='gradient penalty weight (WGAN-GP)')
    args = parser.parse_args()
    main(args)
