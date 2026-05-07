"""compute_fold_stats.py

Run 5-fold CV for Ridge and MLP, record per-fold MSEs, and run paired tests.

Usage:
  python scripts/compute_fold_stats.py --csv data/oasis/OASIS.csv --images data/oasis/Images

This script requires the same data layout as `src/train.py` and will print per-fold
MSEs, means, standard deviations, paired t-test p-values, Wilcoxon p-values,
95% confidence intervals for mean differences, and Cohen's d effect sizes.
"""
import argparse
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

def paired_stats(diffs):
    import math
    from scipy import stats
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    # paired t-test against zero
    t_res = stats.ttest_1samp(diffs, 0.0)
    # Wilcoxon signed-rank (nonparametric)
    try:
        w_res = stats.wilcoxon(diffs)
    except Exception:
        w_res = None
    # 95% CI via t-distribution
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - tcrit * se
    ci_upper = mean + tcrit * se
    cohen_d = mean / sd if sd > 0 else float('inf')
    return {
        'n': n, 'mean': mean, 'sd': sd, 't_stat': float(t_res.statistic), 't_p': float(t_res.pvalue),
        'wilcoxon_stat': float(w_res.statistic) if w_res is not None else None,
        'wilcoxon_p': float(w_res.pvalue) if w_res is not None else None,
        'ci': (ci_lower, ci_upper), 'cohen_d': cohen_d,
    }

def train_mlp(X_train, y_train, X_val, epochs=200, batch=64, lr=1e-3, device=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import build_cnn_model
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_cnn_model(input_dim=X_train.shape[1], output_dim=2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    Xt = torch.from_numpy(X_train).float().to(device)
    Yt = torch.from_numpy(y_train).float().to(device)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch, shuffle=True)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        preds = net(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
    return preds

def main(args):
    # Defer imports that depend on repo layout
    from src.train import build_feature_matrix, get_image_paths, _load_pretrained_classifier
    import pandas as pd

    if not os.path.exists(args.csv) or not os.path.isdir(args.images):
        raise FileNotFoundError('Please provide existing --csv and --images paths.')

    # Build feature matrix for experiment 1
    print('Building features for experiment 1...')
    metadata = pd.read_csv(args.csv)
    image_paths = get_image_paths(args.images)
    X1, yv1, ya1 = build_feature_matrix(image_paths, metadata, experiment=1)

    # Build feature matrix for experiment 2
    print('Building features for experiment 2 (this may be slow)...')
    sem_model, sem_transform = _load_pretrained_classifier()
    X2, yv2, ya2 = build_feature_matrix(image_paths, metadata, experiment=2,
                                        sem_model=sem_model, sem_transform=sem_transform)

    # Helper to run kfold for both models and collect per-fold MSEs
    def kfold_ridge(X, yv, ya):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        v_scores, a_scores = [], []
        for tr, va in kf.split(X):
            vr = Ridge(alpha=1.0).fit(X[tr], yv[tr])
            ar = Ridge(alpha=1.0).fit(X[tr], ya[tr])
            v_scores.append(mean_squared_error(yv[va], vr.predict(X[va])))
            a_scores.append(mean_squared_error(ya[va], ar.predict(X[va])))
        return np.array(v_scores), np.array(a_scores)

    def kfold_mlp(X, yv, ya):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        v_scores, a_scores = [], []
        for tr, va in kf.split(X):
            Ytr = np.stack([yv[tr], ya[tr]], axis=1)
            preds = train_mlp(X[tr], Ytr, X[va], epochs=args.epochs, batch=args.batch, lr=args.lr)
            v_scores.append(mean_squared_error(yv[va], preds[:,0]))
            a_scores.append(mean_squared_error(ya[va], preds[:,1]))
        return np.array(v_scores), np.array(a_scores)

    print('Running Ridge CV for Exp1...')
    r1_v, r1_a = kfold_ridge(X1, yv1, ya1)
    print('Running MLP CV for Exp1...')
    m1_v, m1_a = kfold_mlp(X1, yv1, ya1)

    print('Running Ridge CV for Exp2...')
    r2_v, r2_a = kfold_ridge(X2, yv2, ya2)
    print('Running MLP CV for Exp2...')
    m2_v, m2_a = kfold_mlp(X2, yv2, ya2)

    # Report per-fold and paired stats
    def report(name, r_v, m_v, r_a, m_a):
        print(f'\n=== {name} Valence per-fold MSEs ===')
        print('Ridge:', np.round(r_v, 6))
        print('MLP:  ', np.round(m_v, 6))
        print('Mean (Ridge):', float(r_v.mean()), 'SD:', float(r_v.std(ddof=1)), 'Median:', float(np.median(r_v)))
        print('Mean (MLP):', float(m_v.mean()), 'SD:', float(m_v.std(ddof=1)), 'Median:', float(np.median(m_v)))
        diffs_v = r_v - m_v
        stats_v = paired_stats(diffs_v)
        print('Paired valence stats (Ridge - MLP):', stats_v)

        print(f'\n=== {name} Arousal per-fold MSEs ===')
        print('Ridge:', np.round(r_a, 6))
        print('MLP:  ', np.round(m_a, 6))
        print('Mean (Ridge):', float(r_a.mean()), 'SD:', float(r_a.std(ddof=1)), 'Median:', float(np.median(r_a)))
        print('Mean (MLP):', float(m_a.mean()), 'SD:', float(m_a.std(ddof=1)), 'Median:', float(np.median(m_a)))
        diffs_a = r_a - m_a
        stats_a = paired_stats(diffs_a)
        print('Paired arousal stats (Ridge - MLP):', stats_a)

    report('Experiment 1', r1_v, m1_v, r1_a, m1_a)
    report('Experiment 2', r2_v, m2_v, r2_a, m2_a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
