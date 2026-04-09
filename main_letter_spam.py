# coding=utf-8
# Main — 6개 모델 × 5개 데이터셋 × 3개 지표 (RMSE, MAE, AUROC)

from __future__ import absolute_import, division, print_function

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score

from data_loader import data_loader
from gain import gain                                        # 원본 gain.py 유지
from gain_original_with_history import gain_with_history
from gain_wgan import gain_wgan
from gain_kd import (train_teacher_original, train_teacher_wgan,
                     train_student_kd, inference, count_parameters)
from gain_autoencoder import gain_autoencoder
from gain_missforest import gain_missforest
from utils import normalization, rmse_loss

DATASETS = ['breast', 'spam', 'letter', 'credit', 'news']

MODEL_NAMES = [
    'Original GAIN (Teacher)',
    'WGAN-GP GAIN (Teacher)',
    'Student Original + KD',
    'Student WGAN-GP + KD',
    'AutoEncoder',
    'MissForest',
]

# KD 모델 여부 (파라미터 효율 지표 적용 대상)
IS_KD_MODEL = {
    'Original GAIN (Teacher)' : False,
    'WGAN-GP GAIN (Teacher)'  : False,
    'Student Original + KD'   : True,
    'Student WGAN-GP + KD'    : True,
    'AutoEncoder'             : False,
    'MissForest'              : False,
}


# ══════════════════════════════════════════════════════════════════════
# 평가 지표
# ══════════════════════════════════════════════════════════════════════

def mae_loss(ori_data, imputed_data, data_m):
    ori_norm, norm_params = normalization(ori_data)
    imp_norm, _           = normalization(imputed_data, norm_params)
    mask = (1 - data_m)
    denom = np.sum(mask)
    if denom == 0:
        return 0.0
    return np.sum(np.abs(mask * ori_norm - mask * imp_norm)) / denom


def auroc_loss(ori_data, imputed_data, data_m):
    '''결측 위치의 실제값을 이진화(중앙값 기준)해 AUROC 계산.
    결측 위치가 너무 적거나 클래스가 하나뿐이면 NaN 반환.
    '''
    ori_norm, norm_params = normalization(ori_data)
    imp_norm, _           = normalization(imputed_data, norm_params)
    mask = (1 - data_m).astype(bool)

    y_true_raw = ori_norm[mask]
    y_score    = imp_norm[mask]

    if len(y_true_raw) < 10:
        return float('nan')

    median    = np.median(y_true_raw)
    y_true_bin = (y_true_raw >= median).astype(int)

    if len(np.unique(y_true_bin)) < 2:
        return float('nan')

    try:
        return roc_auc_score(y_true_bin, y_score)
    except Exception:
        return float('nan')


def measure_inference_time(G, norm_data_x, data_m, device, runs=5):
    from utils import uniform_sampler
    no, dim = norm_data_x.shape
    times   = []
    for _ in range(runs):
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        X_mb = data_m * norm_data_x + (1 - data_m) * Z_mb
        Xt   = torch.tensor(X_mb, dtype=torch.float32).to(device)
        Mt   = torch.tensor(data_m, dtype=torch.float32).to(device)
        t0   = time.time()
        with torch.no_grad(): G(Xt, Mt)
        times.append((time.time() - t0) * 1000)
    return float(np.mean(times))


# ══════════════════════════════════════════════════════════════════════
# 출력: 데이터셋별 RMSE/MAE 상세 테이블
# ══════════════════════════════════════════════════════════════════════

def print_per_dataset_table(all_results, metric):
    '''각 모델의 데이터셋별 RMSE 또는 MAE 출력.

    형식: Model | breast | spam | letter | credit | news | avg
    '''
    print(f'\n{"─"*80}')
    print(f'  {metric.upper()} — 데이터셋별 상세 결과')
    print(f'{"─"*80}')
    header = f"  {'Model':28s}" + ''.join(f'{d:>9s}' for d in DATASETS) + f"{'avg':>9s}"
    print(header)
    print(f'{"─"*80}')

    for model in MODEL_NAMES:
        vals = []
        for ds in DATASETS:
            v = all_results.get(model, {}).get(ds, {}).get(metric, float('nan'))
            vals.append(v)
        avg    = np.nanmean(vals)
        row    = f"  {model:28s}"
        row   += ''.join(f'{v:9.4f}' if not np.isnan(v) else f"{'N/A':>9s}" for v in vals)
        row   += f'{avg:9.4f}'
        print(row)
    print(f'{"─"*80}')


# ══════════════════════════════════════════════════════════════════════
# 출력: 종합 성능 테이블 (avg RMSE, MAE, AUROC + 파라미터 효율)
# ══════════════════════════════════════════════════════════════════════

def print_summary_table(all_results, param_info):
    print(f'\n{"═"*95}')
    print(f"  {'Model':28s} {'RMSE(avg)':>10s} {'MAE(avg)':>10s} {'AUROC(avg)':>11s} "
          f"{'Params':>9s} {'Infer(ms)':>10s} {'Eff(RMSE/P)':>12s}")
    print(f'{"═"*95}')

    # Teacher RMSE 기준 (Original GAIN Teacher, 전체 데이터셋 평균)
    base_rmse = np.nanmean([
        all_results.get('Original GAIN (Teacher)', {}).get(ds, {}).get('rmse', np.nan)
        for ds in DATASETS
    ])

    for model in MODEL_NAMES:
        rmse_vals  = [all_results.get(model, {}).get(ds, {}).get('rmse',  np.nan) for ds in DATASETS]
        mae_vals   = [all_results.get(model, {}).get(ds, {}).get('mae',   np.nan) for ds in DATASETS]
        auroc_vals = [all_results.get(model, {}).get(ds, {}).get('auroc', np.nan) for ds in DATASETS]

        avg_rmse  = np.nanmean(rmse_vals)
        avg_mae   = np.nanmean(mae_vals)
        avg_auroc = np.nanmean(auroc_vals)

        pinfo     = param_info.get(model, {})
        params    = pinfo.get('params', 0)
        infer_ms  = pinfo.get('infer_ms', 0.0)

        # 파라미터 효율 지표: RMSE 개선율(%) / 파라미터 감소율(%)
        # KD 모델에만 적용
        if IS_KD_MODEL[model] and params > 0:
            teacher_key = ('Original GAIN (Teacher)' if 'Original' in model
                           else 'WGAN-GP GAIN (Teacher)')
            t_params  = param_info.get(teacher_key, {}).get('params', params)
            t_rmse    = np.nanmean([
                all_results.get(teacher_key, {}).get(ds, {}).get('rmse', np.nan)
                for ds in DATASETS
            ])
            param_red = (t_params - params) / t_params * 100   # 파라미터 감소율(%)
            rmse_chg  = (t_rmse - avg_rmse) / t_rmse * 100     # RMSE 개선율(%) — 양수=개선
            eff_str   = f'{rmse_chg/param_red*100:.3f}' if param_red > 0 else 'N/A'
        else:
            eff_str = '—'

        params_str = f'{params:,}' if params > 0 else 'N/A'
        infer_str  = f'{infer_ms:.2f}' if infer_ms > 0 else 'N/A'

        print(f"  {model:28s} {avg_rmse:>10.4f} {avg_mae:>10.4f} {avg_auroc:>11.4f} "
              f"{params_str:>9s} {infer_str:>10s} {eff_str:>12s}")
    print(f'{"═"*95}')
    print('  * Eff(RMSE/P): RMSE개선율 / 파라미터감소율 × 100  (KD 모델만 적용)')


# ══════════════════════════════════════════════════════════════════════
# 그래프
# ══════════════════════════════════════════════════════════════════════

def plot_results(all_results, param_info, save_path):
    n_ds  = len(DATASETS)
    n_mod = len(MODEL_NAMES)
    colors = ['steelblue','tomato','mediumseagreen','darkorange','mediumpurple','saddlebrown']

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('GAIN Family — 6 Models × 5 Datasets × 3 Metrics',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    x    = np.arange(n_ds)
    w    = 0.13
    offs = np.linspace(-(n_mod-1)/2*w, (n_mod-1)/2*w, n_mod)

    def bar_group(ax, metric, title, ylabel):
        for i, (model, color) in enumerate(zip(MODEL_NAMES, colors)):
            vals = [all_results.get(model, {}).get(ds, {}).get(metric, np.nan)
                    for ds in DATASETS]
            ax.bar(x + offs[i], vals, width=w, label=model, color=color, alpha=0.85)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    # ── 1. RMSE per dataset ──────────────────────────────────────
    bar_group(fig.add_subplot(gs[0, 0]), 'rmse', 'RMSE per Dataset', 'RMSE ↓')

    # ── 2. MAE per dataset ───────────────────────────────────────
    bar_group(fig.add_subplot(gs[0, 1]), 'mae',  'MAE per Dataset',  'MAE ↓')

    # ── 3. AUROC per dataset ─────────────────────────────────────
    bar_group(fig.add_subplot(gs[0, 2]), 'auroc','AUROC per Dataset','AUROC ↑')

    # ── 4. Average RMSE 비교 ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    avg_rmse = [np.nanmean([all_results.get(m,{}).get(ds,{}).get('rmse',np.nan)
                            for ds in DATASETS]) for m in MODEL_NAMES]
    bars = ax4.bar(range(n_mod), avg_rmse, color=colors, width=0.5)
    ax4.set_xticks(range(n_mod))
    ax4.set_xticklabels([m.replace(' ', '\n') for m in MODEL_NAMES], fontsize=6)
    ax4.set_title('Avg RMSE (all datasets)'); ax4.set_ylabel('RMSE ↓')
    for bar, v in zip(bars, avg_rmse):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    # ── 5. Average MAE 비교 ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    avg_mae = [np.nanmean([all_results.get(m,{}).get(ds,{}).get('mae',np.nan)
                           for ds in DATASETS]) for m in MODEL_NAMES]
    bars = ax5.bar(range(n_mod), avg_mae, color=colors, width=0.5)
    ax5.set_xticks(range(n_mod))
    ax5.set_xticklabels([m.replace(' ', '\n') for m in MODEL_NAMES], fontsize=6)
    ax5.set_title('Avg MAE (all datasets)'); ax5.set_ylabel('MAE ↓')
    for bar, v in zip(bars, avg_mae):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax5.grid(True, alpha=0.3, axis='y')

    # ── 6. Average AUROC 비교 ────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    avg_auroc = [np.nanmean([all_results.get(m,{}).get(ds,{}).get('auroc',np.nan)
                             for ds in DATASETS]) for m in MODEL_NAMES]
    bars = ax6.bar(range(n_mod), avg_auroc, color=colors, width=0.5)
    ax6.set_xticks(range(n_mod))
    ax6.set_xticklabels([m.replace(' ', '\n') for m in MODEL_NAMES], fontsize=6)
    ax6.set_title('Avg AUROC (all datasets)'); ax6.set_ylabel('AUROC ↑')
    for bar, v in zip(bars, avg_auroc):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax6.grid(True, alpha=0.3, axis='y')

    # ── 7. 파라미터 수 비교 (KD 모델 포함 전체) ──────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    p_models = [m for m in MODEL_NAMES if param_info.get(m, {}).get('params', 0) > 0]
    p_vals   = [param_info[m]['params'] for m in p_models]
    p_colors = [colors[MODEL_NAMES.index(m)] for m in p_models]
    ax7.bar(range(len(p_models)), p_vals, color=p_colors, width=0.5)
    ax7.set_xticks(range(len(p_models)))
    ax7.set_xticklabels([m.replace(' ','\n') for m in p_models], fontsize=6)
    ax7.set_title('Model Parameters'); ax7.set_ylabel('# Parameters ↓')
    ax7.grid(True, alpha=0.3, axis='y')

    # ── 8. KD Teacher vs Student 비교 (RMSE) ─────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    pairs = [('Original GAIN (Teacher)', 'Student Original + KD'),
             ('WGAN-GP GAIN (Teacher)',  'Student WGAN-GP + KD')]
    pair_labels = ['Orig Teacher', 'Orig Student', 'WGAN Teacher', 'WGAN Student']
    pair_vals   = []
    pair_colors = []
    for t, s in pairs:
        pair_vals.append(np.nanmean([all_results.get(t,{}).get(ds,{}).get('rmse',np.nan) for ds in DATASETS]))
        pair_vals.append(np.nanmean([all_results.get(s,{}).get(ds,{}).get('rmse',np.nan) for ds in DATASETS]))
        pair_colors += [colors[MODEL_NAMES.index(t)], colors[MODEL_NAMES.index(s)]]
    ax8.bar(range(4), pair_vals, color=pair_colors, width=0.5)
    ax8.set_xticks(range(4)); ax8.set_xticklabels(pair_labels, fontsize=7)
    ax8.set_title('KD: Teacher vs Student\n(Avg RMSE)'); ax8.set_ylabel('RMSE ↓')
    for i, v in enumerate(pair_vals):
        ax8.text(i, v+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax8.grid(True, alpha=0.3, axis='y')

    # ── 9. Summary 텍스트 ────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    lines = ['── Summary (avg across datasets) ──\n']
    for model in MODEL_NAMES:
        rmse_v  = np.nanmean([all_results.get(model,{}).get(ds,{}).get('rmse',np.nan) for ds in DATASETS])
        mae_v   = np.nanmean([all_results.get(model,{}).get(ds,{}).get('mae',np.nan)  for ds in DATASETS])
        auroc_v = np.nanmean([all_results.get(model,{}).get(ds,{}).get('auroc',np.nan) for ds in DATASETS])
        lines.append(f'{model[:22]:22s}')
        lines.append(f'  RMSE:{rmse_v:.4f} MAE:{mae_v:.4f} AUC:{auroc_v:.4f}\n')

    ax9.text(0.02, 0.98, ''.join(lines), transform=ax9.transAxes,
             fontsize=7.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n[그래프 저장] → {save_path}')


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def evaluate(ori_data, imputed_data, data_m):
    return {
        'rmse' : rmse_loss(ori_data, imputed_data, data_m),
        'mae'  : mae_loss(ori_data, imputed_data, data_m),
        'auroc': auroc_loss(ori_data, imputed_data, data_m),
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        'batch_size': args.batch_size,
        'hint_rate' : args.hint_rate,
        'alpha'     : args.alpha,
        'iterations': args.iterations,
        'n_critic'  : args.n_critic,
        'lambda_gp' : args.lambda_gp,
    }

    # all_results[model][dataset] = {rmse, mae, auroc}
    all_results = {m: {} for m in MODEL_NAMES}
    param_info  = {m: {} for m in MODEL_NAMES}

    for ds in DATASETS:
        print(f'\n{"#"*65}')
        print(f'#  데이터셋: {ds.upper()}')
        print(f'{"#"*65}')

        ori_data_x, miss_data_x, data_m = data_loader(ds, args.miss_rate)

        # ── [1] Original GAIN Teacher ─────────────────────────────
        print(f'\n[{ds}] 1/6  Original GAIN Teacher')
        T_orig, nd_orig, dm_orig, np_orig = train_teacher_original(miss_data_x, params, device)
        imp = inference(T_orig, nd_orig, dm_orig, np_orig, miss_data_x, device)
        all_results['Original GAIN (Teacher)'][ds] = evaluate(ori_data_x, imp, data_m)
        if ds == DATASETS[0]:
            param_info['Original GAIN (Teacher)'] = {
                'params'  : count_parameters(T_orig),
                'infer_ms': measure_inference_time(T_orig, nd_orig, dm_orig, device),
            }

        # ── [2] WGAN-GP Teacher ───────────────────────────────────
        print(f'\n[{ds}] 2/6  WGAN-GP GAIN Teacher')
        T_wgan, nd_wgan, dm_wgan, np_wgan = train_teacher_wgan(miss_data_x, params, device)
        imp = inference(T_wgan, nd_wgan, dm_wgan, np_wgan, miss_data_x, device)
        all_results['WGAN-GP GAIN (Teacher)'][ds] = evaluate(ori_data_x, imp, data_m)
        if ds == DATASETS[0]:
            param_info['WGAN-GP GAIN (Teacher)'] = {
                'params'  : count_parameters(T_wgan),
                'infer_ms': measure_inference_time(T_wgan, nd_wgan, dm_wgan, device),
            }

        # ── [3] Student Original + KD ─────────────────────────────
        print(f'\n[{ds}] 3/6  Student Original GAIN + KD (0.5x)')
        S_orig, nd_s, dm_s, np_s, _, _ = train_student_kd(
            miss_data_x, T_orig, params, use_wgan=False, device=device,
            temperature=args.temperature, kd_weight=args.kd_weight)
        imp = inference(S_orig, nd_s, dm_s, np_s, miss_data_x, device)
        all_results['Student Original + KD'][ds] = evaluate(ori_data_x, imp, data_m)
        if ds == DATASETS[0]:
            param_info['Student Original + KD'] = {
                'params'  : count_parameters(S_orig),
                'infer_ms': measure_inference_time(S_orig, nd_s, dm_s, device),
            }

        # ── [4] Student WGAN-GP + KD ──────────────────────────────
        print(f'\n[{ds}] 4/6  Student WGAN-GP GAIN + KD (0.5x)')
        S_wgan, nd_sw, dm_sw, np_sw, _, _ = train_student_kd(
            miss_data_x, T_wgan, params, use_wgan=True, device=device,
            temperature=args.temperature, kd_weight=args.kd_weight)
        imp = inference(S_wgan, nd_sw, dm_sw, np_sw, miss_data_x, device)
        all_results['Student WGAN-GP + KD'][ds] = evaluate(ori_data_x, imp, data_m)
        if ds == DATASETS[0]:
            param_info['Student WGAN-GP + KD'] = {
                'params'  : count_parameters(S_wgan),
                'infer_ms': measure_inference_time(S_wgan, nd_sw, dm_sw, device),
            }

        # ── [5] AutoEncoder ───────────────────────────────────────
        print(f'\n[{ds}] 5/6  AutoEncoder')
        imp = gain_autoencoder(miss_data_x, params)
        all_results['AutoEncoder'][ds] = evaluate(ori_data_x, imp, data_m)

        # ── [6] MissForest ────────────────────────────────────────
        print(f'\n[{ds}] 6/6  MissForest')
        imp = gain_missforest(miss_data_x, params)
        all_results['MissForest'][ds] = evaluate(ori_data_x, imp, data_m)

        # 데이터셋별 중간 결과 출력
        print(f'\n  [{ds.upper()} 결과 요약]')
        for model in MODEL_NAMES:
            m = all_results[model][ds]
            print(f"    {model:30s}  RMSE:{m['rmse']:.4f}  MAE:{m['mae']:.4f}  "
                  f"AUROC:{m['auroc']:.4f}" if not np.isnan(m['auroc'])
                  else f"    {model:30s}  RMSE:{m['rmse']:.4f}  MAE:{m['mae']:.4f}  AUROC: N/A")

    # ── 원본 gain.py 호출 확인 ────────────────────────────────────
    print('\n[원본 gain.py 직접 호출 확인]')
    ori_data_x, miss_data_x, data_m = data_loader('spam', args.miss_rate)
    orig_check = gain(miss_data_x, params)
    check_rmse = rmse_loss(ori_data_x, orig_check, data_m)
    print(f'  gain.py RMSE (spam): {check_rmse:.4f}')

    # ── 최종 결과 출력 ────────────────────────────────────────────
    print_per_dataset_table(all_results, 'rmse')
    print_per_dataset_table(all_results, 'mae')
    print_summary_table(all_results, param_info)

    # ── 그래프 저장 ───────────────────────────────────────────────
    plot_results(all_results, param_info, save_path='results_full.png')

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--miss_rate',   default=0.2,   type=float)
    parser.add_argument('--batch_size',  default=128,   type=int)
    parser.add_argument('--hint_rate',   default=0.9,   type=float)
    parser.add_argument('--alpha',       default=100,   type=float)
    parser.add_argument('--iterations',  default=10000, type=int)
    parser.add_argument('--n_critic',    default=5,     type=int)
    parser.add_argument('--lambda_gp',   default=10,    type=float)
    parser.add_argument('--temperature', default=2.0,   type=float)
    parser.add_argument('--kd_weight',   default=0.5,   type=float)
    args = parser.parse_args()
    main(args)
