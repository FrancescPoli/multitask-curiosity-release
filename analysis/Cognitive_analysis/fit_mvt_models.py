"""
Fit MVT GLM to Model Behavior (Grid-Search Recovery)
=====================================================

For each model run, searches over (alpha, beta) EMA rate pairs to find the
timescales that best explain the agent's stay/leave decisions, then fits a
logistic regression at the best grid point:

    P(Leave) ~ Bernoulli(logit = b0 + b1 * (P - rho))

Key implementation details:

* EMAs are computed on the **full step stream** (training + travel), with
  travel rewards filled in as zeros. Travel rows are sub-sampled in the
  log (~every 100 steps), so missing step indices between two logged rows
  are imputed to zero — exactly what the agent observed internally.
* The GLM is fit only on *decision* rows (mode == 'train', decision ∈ {0,1}).
* Subsamples to MAX_SAMPLES (50k) decision rows for the GLM fit.
* Log-spaced grid spans timescales from 1 step up to ~7000 steps so the
  configured agent half-lives (69, 231, 693, 6931) are recoverable.

Usage:
    python analysis/Cognitive_analysis/fit_mvt_models.py --sweep_dir Z:/...
"""

import pandas as pd
import numpy as np
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from pathlib import Path
import argparse
import re
import sys


# --- Grid definition (log-spaced) ---
# Covers alpha_hl up to 100, beta_hl up to 7000 — spans 4 decades so the
# configured agent half-lives (3.5, 6.9, 23, 69 for alpha; 69, 231, 693, 6931
# for beta) all have a near-exact grid point.
HALF_LIVES_P = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]
HALF_LIVES_RHO = [
    1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100,
    150, 230, 350, 500, 700, 1000, 1500, 2300, 3500, 5000, 7000,
]

# Constrain global half-life to be at least MIN_RATIO x local half-life.
# MIN_RATIO = 1 allows the alpha == beta diagonal.
MIN_RATIO = 1

MAX_SAMPLES = 50000  # Subsample decision rows to this many for GLM fitting


def alpha_from_hl(hl):
    """Convert half-life (in steps) to EMA rate."""
    return np.log(2) / hl


@njit
def ema_all_rates(rewards, rates):
    """EMA for all rates at once. Returns (n_samples, n_rates) matrix."""
    n = len(rewards)
    k = len(rates)
    out = np.empty((n, k))
    for j in range(k):
        x = 0.0
        r = rates[j]
        for i in range(n):
            x = (1.0 - r) * x + r * rewards[i]
            out[i, j] = x
    return out


def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def parse_run_params(run_dir_name):
    params = {}
    patterns = {
        'l1': r'_l1-([0-9e\-\.]+)',
        'dist': r'_dist-([0-9e\-\.]+)',
        'alpha': r'_alpha([0-9\.]+)',
        'beta': r'_beta([0-9\.]+)',
        'temp': r'_temp([0-9\.]+)',
        'trav': r'_trav(\d+)',
        'eps': r'_eps([0-9\.\-]+)',
    }
    for key, pat in patterns.items():
        match = re.search(pat, run_dir_name)
        if match:
            try:
                params[key] = float(match.group(1))
            except ValueError:
                pass
    return params


def build_dense_stream(df):
    """Reconstruct the full per-step reward stream.

    The curriculum log subsamples travel rows (~every 100 steps), so step
    indices between two logged rows are missing. Since those missing steps
    are all zero-reward travel steps (what the agent observed), we fill them
    with zero by initialising a dense array and scattering logged rewards
    into it at their step index.
    """
    # Steps are global_step counters, strictly increasing.
    df = df.sort_values('step').reset_index(drop=True)
    max_step = int(df['step'].iloc[-1])

    dense = np.zeros(max_step + 1, dtype=np.float64)
    step_idx = df['step'].values.astype(np.int64)
    rewards = df['reward'].fillna(0.0).values.astype(np.float64)
    dense[step_idx] = rewards
    return dense, step_idx


def fit_mvt_glm(run_dir):
    run_path = Path(run_dir)
    params = parse_run_params(run_path.name)

    curriculum_file = run_path / "run_curriculum.csv"
    if not curriculum_file.exists():
        return None

    try:
        df = pd.read_csv(
            curriculum_file,
            usecols=['step', 'mode', 'reward', 'decision'],
            on_bad_lines='skip',
        )
        df = df.dropna(subset=['step']).copy()
        df['step'] = df['step'].astype(np.int64)

        if len(df) < 50:
            return None

        # 1) Reconstruct dense per-step reward stream (zeros on missing travel rows)
        dense_rewards, _ = build_dense_stream(df)

        # 2) Compute EMAs on the FULL stream at every rate
        alpha_rates = np.array([alpha_from_hl(hl) for hl in HALF_LIVES_P])
        beta_rates = np.array([alpha_from_hl(hl) for hl in HALF_LIVES_RHO])
        P_full = ema_all_rates(dense_rewards, alpha_rates)     # (n_steps, n_alphas)
        rho_full = ema_all_rates(dense_rewards, beta_rates)    # (n_steps, n_betas)

        # 3) Index EMAs at decision rows only (mode == 'train' and decision in {0,1})
        is_train = df['mode'].astype(str).str.lower() == 'train'
        has_decision = df['decision'].isin([0, 1])
        decision_mask = (is_train & has_decision).values
        if decision_mask.sum() < 50:
            return None

        dec_steps = df.loc[decision_mask, 'step'].values.astype(np.int64)
        y_all = df.loc[decision_mask, 'decision'].values.astype(int)
        P_dec = P_full[dec_steps]      # (n_decisions, n_alphas)
        rho_dec = rho_full[dec_steps]  # (n_decisions, n_betas)
        n_total = len(y_all)

        # 4) Subsample decision rows for the GLM fit (keeps total <= MAX_SAMPLES)
        if n_total > MAX_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_total, MAX_SAMPLES, replace=False)
            idx.sort()
            y_sub = y_all[idx]
            P_sub = P_dec[idx]
            rho_sub = rho_dec[idx]
        else:
            y_sub = y_all
            P_sub = P_dec
            rho_sub = rho_dec

        best_aic = np.inf
        best_result = None

        for i_a in range(len(alpha_rates)):
            P_col = P_sub[:, i_a]
            a_hl = HALF_LIVES_P[i_a]

            for i_b in range(len(beta_rates)):
                b_hl = HALF_LIVES_RHO[i_b]

                if b_hl < MIN_RATIO * a_hl:
                    continue

                rho_col = rho_sub[:, i_b]
                X_raw = P_col - rho_col
                std_X = float(np.std(X_raw))

                if std_X < 1e-12:
                    continue

                X = X_raw.reshape(-1, 1)

                try:
                    clf = LogisticRegression(
                        fit_intercept=True, C=1e9, solver='lbfgs', max_iter=200
                    )
                    clf.fit(X, y_sub)

                    probs = clf.predict_proba(X)
                    ll = -log_loss(y_sub, probs, normalize=False)
                    aic = 2 * 2 - 2 * ll

                    if aic < best_aic:
                        best_aic = aic
                        b1 = float(clf.coef_[0][0])
                        b0 = float(clf.intercept_[0])

                        baseline_stay = float(sigmoid(-b0))
                        p_stay_high = float(sigmoid(-(b0 + b1 * std_X)))
                        p_stay_low = float(sigmoid(-(b0 - b1 * std_X)))
                        reward_modulation = float(
                            np.clip(p_stay_high - p_stay_low, 0.0, 1.0)
                        )
                        timescale_balance = a_hl / (a_hl + b_hl)

                        if b1 != 0:
                            T_est = -1.0 / b1
                            eps_est = b0 * T_est
                        else:
                            T_est = np.nan
                            eps_est = np.nan

                        best_result = {
                            'run_id': run_path.name,
                            'baseline_stay': baseline_stay,
                            'reward_modulation': reward_modulation,
                            'timescale_balance': timescale_balance,
                            'fitted_alpha_hl': a_hl,
                            'fitted_beta_hl': b_hl,
                            'fitted_T': T_est,
                            'fitted_eps': eps_est,
                            'beta0': b0,
                            'beta1': b1,
                            'std_X': std_X,
                            'aic': best_aic,
                            'valid_mvt': bool(b1 < 0),
                            'n_decisions': int(n_total),
                            'n_fit': int(len(y_sub)),
                            'n_steps_total': int(len(dense_rewards)),
                            'config_alpha': params.get('alpha'),
                            'config_beta': params.get('beta'),
                            'config_temp': params.get('temp'),
                            'config_l1': params.get('l1'),
                            'config_trav': params.get('trav'),
                        }
                except Exception:
                    continue

        return best_result

    except Exception as e:
        print(f"Error fitting {run_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fit MVT GLM to Model Traces (Grid-Search Recovery)"
    )
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument(
        "--output", type=str,
        default="analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv"
    )
    parser.add_argument("--n_shards", type=int, default=1,
                        help="Split run list across this many shards (for parallel jobs).")
    parser.add_argument("--shard", type=int, default=0,
                        help="Which shard this job handles (0-indexed, < n_shards).")
    parser.add_argument("--merge", action="store_true",
                        help="Concatenate all shard CSVs matching --output into a single "
                             "file and exit. Use after all shard jobs complete.")

    args = parser.parse_args()
    sweep_path = Path(args.sweep_dir)
    output_path = Path(args.output)

    if args.merge:
        pattern = f"{output_path.stem}_shard*of*{output_path.suffix}"
        shard_files = sorted(output_path.parent.glob(pattern))
        if not shard_files:
            print(f"No shard CSVs found matching {output_path.parent / pattern}")
            sys.exit(1)
        dfs = [pd.read_csv(p) for p in shard_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"Merged {len(shard_files)} shards ({len(merged)} rows) -> {output_path}")
        sys.exit(0)

    if args.n_shards > 1:
        if not (0 <= args.shard < args.n_shards):
            print(f"--shard must be in [0, {args.n_shards}); got {args.shard}")
            sys.exit(1)
        output_path = output_path.with_name(
            f"{output_path.stem}_shard{args.shard}of{args.n_shards}{output_path.suffix}"
        )

    if not sweep_path.exists():
        print(f"Directory not found: {sweep_path}")
        sys.exit(1)

    print(f"Scanning {sweep_path}...")
    run_dirs = sorted([
        d for d in sweep_path.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    print(f"Found {len(run_dirs)} runs.")

    if args.n_shards > 1:
        run_dirs = run_dirs[args.shard::args.n_shards]
        print(f"Shard {args.shard}/{args.n_shards}: processing {len(run_dirs)} runs "
              f"-> {output_path.name}")

    processed_ids = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if 'run_id' in existing_df.columns:
                processed_ids = set(existing_df['run_id'].unique())
                print(f"Resuming: Found {len(processed_ids)} already processed.")
        except Exception:
            pass

    total = len(run_dirs)
    count = 0

    for i, run_dir in enumerate(run_dirs):
        if run_dir.name in processed_ids:
            continue

        if i % 5 == 0 or i == total - 1:
            print(
                f"Processing {i+1}/{total}: {run_dir.name}...",
                end='\r', flush=True
            )

        res = fit_mvt_glm(run_dir)

        if res:
            df_out = pd.DataFrame([res])
            header = not output_path.exists()
            df_out.to_csv(output_path, mode='a', header=header, index=False)
            count += 1

    print(f"\nProcessing complete. Fitted {count} new models.")


if __name__ == "__main__":
    main()
