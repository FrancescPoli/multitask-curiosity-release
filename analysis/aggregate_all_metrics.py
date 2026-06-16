"""
Grand Unified Metrics Aggregation
=================================

Aggregates metrics from four analysis domains into a single master CSV:
1. Synaptic Analysis: `analysis/Synaptic_analysis/Results/synaptic_metrics.csv`
2. Network Analysis: `analysis/Network_analysis/Results/rnn_topological_metrics.csv`
3. Cognitive Analysis: `analysis/Cognitive_analysis/Results/cognitive_metrics.csv`
4. Performance Analysis: `analysis/Performance_analysis/Results/performance_metrics.csv`

Computes Euclidean distances to human baselines in normalized space.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

# --- Configuration ---
SYNAPTIC_CSV = "analysis/Synaptic_analysis/Results/synaptic_metrics.csv"
NETWORK_CSV = "analysis/Network_analysis/Results/rnn_topological_metrics.csv"
COGNITIVE_CSV = "analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv"
PERFORMANCE_CSV = "analysis/Performance_analysis/Results/performance_metrics.csv"
INFANT_PARAMS_CSV = "analysis/Cognitive_analysis/Results/infant_mvt_params.csv"
HCP_METRICS_CSV = "analysis/Network_analysis/Results/human_topological_metrics.csv"

OUTPUT_CSV = "analysis/grand_unified_metrics_v2.csv"

# Human Synaptic (Hardcoded references from Huttenlocher)
HUMAN_SYN_PEAK = 3.24
HUMAN_SYN_DECAY = 0.0310
HUMAN_SYN_CORR = 1.0
HUMAN_SYN_PEAK_FRACTION = 3.24 / 75  # peak_age / max_age_cap — ~0.043

def load_and_validate(path):
    p = Path(path)
    if not p.exists():
        print(f"Warning: File not found: {p}")
        return None
    return pd.read_csv(p)

def main():
    print("Aggregating Grand Unified Metrics...")

    # 1. Load Data
    syn_df = load_and_validate(SYNAPTIC_CSV)
    net_df = load_and_validate(NETWORK_CSV)
    cog_df = load_and_validate(COGNITIVE_CSV)
    perf_df = load_and_validate(PERFORMANCE_CSV)

    hcp_df = load_and_validate(HCP_METRICS_CSV)
    inf_df = load_and_validate(INFANT_PARAMS_CSV)

    if syn_df is None:
        print("Critical Error: Synaptic CSV missing. Cannot proceed.")
        return

    # 2. Prepare Human Baselines
    # Network
    if hcp_df is not None:
        # Use mean of all HCPya individual subjects as the human reference
        hcpya = hcp_df[hcp_df['dataset'] == 'HCPya']
        hum_mod = hcpya['modularity'].mean()
        hum_eff = hcpya['efficiency'].mean()
        hum_rc  = hcpya['rich_club'].mean()

        hum_mod_sd = hcpya['modularity'].std()
        hum_eff_sd = hcpya['efficiency'].std()
        hum_rc_sd  = hcpya['rich_club'].std()
    else:
        hum_mod = hum_eff = hum_rc = np.nan
        hum_mod_sd = hum_eff_sd = hum_rc_sd = 1.0

    # Cognitive — compute bounded [0,1] metrics for infants
    if inf_df is not None:
        def _sigmoid(x):
            x = np.clip(x, -20, 20)
            return 1.0 / (1.0 + np.exp(-x))

        # Filter invalid fits (beta1 must be negative for valid MVT)
        inf_valid = inf_df[inf_df['valid_mvt'] == True].copy() if 'valid_mvt' in inf_df.columns else inf_df.copy()

        # Compute bounded metrics for each infant
        inf_valid['baseline_stay'] = _sigmoid(-inf_valid['beta0'])

        # Reward modulation: P(stay|+1SD) - P(stay|-1SD)
        inf_valid['reward_modulation'] = (
            _sigmoid(-(inf_valid['beta0'] + inf_valid['beta1'] * inf_valid['signal_sd']))
            - _sigmoid(-(inf_valid['beta0'] - inf_valid['beta1'] * inf_valid['signal_sd']))
        ).clip(lower=0.0, upper=1.0)

        # Timescale balance: alpha_hl / (alpha_hl + beta_hl)
        inf_valid['timescale_balance'] = inf_valid['alpha_hl'] / (inf_valid['alpha_hl'] + inf_valid['beta_hl'])

        # Clean: remove outliers (4SD on each bounded metric)
        for col in ['baseline_stay', 'reward_modulation', 'timescale_balance']:
            m = inf_valid[col].mean()
            s = inf_valid[col].std()
            if s > 0:
                inf_valid = inf_valid[abs(inf_valid[col] - m) <= 4 * s]

        # Infant group means (the target point — robust at group level)
        hum_baseline_stay = inf_valid['baseline_stay'].mean()
        hum_reward_mod = inf_valid['reward_modulation'].mean()
        hum_timescale_bal = inf_valid['timescale_balance'].mean()
    else:
        hum_baseline_stay = hum_reward_mod = hum_timescale_bal = np.nan


    # 3. Merge Data (Left join on Synaptic to keep all runs)
    # Synaptic
    merged = syn_df.copy()
    
    # Network
    if net_df is not None:
        # Drop params if they exist to avoid dups
        cols = [c for c in net_df.columns if c not in merged.columns or c == 'run_id']
        merged = pd.merge(merged, net_df[cols], on='run_id', how='left')
        
    # Cognitive
    if cog_df is not None:
        # Drop dups
        cols = [c for c in cog_df.columns if c not in merged.columns or c == 'run_id']
        merged = pd.merge(merged, cog_df[cols], on='run_id', how='left')

    # Performance
    if perf_df is not None:
        cols = [c for c in perf_df.columns if c not in merged.columns or c == 'run_id']
        merged = pd.merge(merged, perf_df[cols], on='run_id', how='left')

    # 4. Compute Distances (Z-scored)
    
    # --- New Logic: Parse Parameters & Refine Data Structure (Part A) ---
    def parse_params(row):
        run_id = str(row['run_id'])
        params = {}
        
        # Regularization Type & Value
        if '_dist-' in run_id:
            params['reg_type'] = 'Distance'
            # Extract value: ..._dist-1e-05_...
            try:
                # Find part starting with dist-
                parts = run_id.split('_')
                for p in parts:
                    if p.startswith('dist-'):
                        params['reg_value'] = float(p.split('-', 1)[1])
                        break
            except:
                params['reg_value'] = np.nan
        elif '_l1-' in run_id:
            params['reg_type'] = 'L1'
            try:
                parts = run_id.split('_')
                for p in parts:
                    if p.startswith('l1-'):
                        params['reg_value'] = float(p.split('-', 1)[1])
                        break
            except:
                params['reg_value'] = np.nan
        else:
            params['reg_type'] = 'None'
            params['reg_value'] = 0.0
            
        return pd.Series(params)

    # Apply parsing
    metric_cols = merged.apply(parse_params, axis=1)
    merged = pd.concat([merged, metric_cols], axis=1)

    # Parse eps from run_id if not already present (omitted in dirname when 0.0)
    if 'eps' not in merged.columns:
        def _parse_eps(run_id):
            m = re.search(r'eps(-?[\d\.e-]+)', str(run_id))
            return float(m.group(1)) if m else 0.0
        merged['eps'] = merged['run_id'].apply(_parse_eps)
    
    # --------------------------------------------------------------------

    # Synaptic (Already computed in R, normalized)
    # R distance = sqrt((1-cor)^2 + ((peak-ref)/40)^2 + ((decay-ref)/.1)^2)
    # We rename it.
    merged.rename(columns={'distance': 'dist_synaptic'}, inplace=True)
    
    # Network
    # Z-score relative to HUMAN distribution
    merged['z_mod'] = (merged['modularity'] - hum_mod) / hum_mod_sd
    merged['z_eff'] = (merged['efficiency'] - hum_eff) / hum_eff_sd
    merged['z_rc']  = (merged['rich_club'] - hum_rc) / hum_rc_sd
    
    # Euclidean distance in Z-space. Missing values -> NaN distance.
    merged['dist_network'] = np.sqrt(
        merged['z_mod']**2 + 
        merged['z_eff']**2 + 
        merged['z_rc']**2
    )
    
    # Cognitive — raw Euclidean distance in bounded [0,1] space (no z-scoring)
    merged['dist_cognitive'] = np.sqrt(
        (merged['baseline_stay'] - hum_baseline_stay)**2 +
        (merged['reward_modulation'] - hum_reward_mod)**2 +
        (merged['timescale_balance'] - hum_timescale_bal)**2
    )

    # Grand Distance: Normalize each subspace to [0,1] via 95th percentile, then L2.
    q95_syn = merged['dist_synaptic'].quantile(0.95)
    q95_net = merged['dist_network'].quantile(0.95)
    q95_cog = merged['dist_cognitive'].quantile(0.95)

    merged['norm_dist_synaptic']  = (merged['dist_synaptic']  / q95_syn).clip(upper=1.0)
    merged['norm_dist_network']   = (merged['dist_network']   / q95_net).clip(upper=1.0)
    merged['norm_dist_cognitive'] = (merged['dist_cognitive']  / q95_cog).clip(upper=1.0)

    merged['grand_distance'] = np.sqrt(
        merged['norm_dist_synaptic']**2 +
        merged['norm_dist_network']**2 +
        merged['norm_dist_cognitive']**2
    )

    # 5. Add Human Baseline Columns (Repeated)
    merged['human_syn_peak'] = HUMAN_SYN_PEAK
    merged['human_syn_decay'] = HUMAN_SYN_DECAY
    merged['human_syn_corr'] = HUMAN_SYN_CORR
    
    merged['human_net_modularity'] = hum_mod
    merged['human_net_efficiency'] = hum_eff
    merged['human_net_rich_club'] = hum_rc
    
    merged['human_cog_baseline_stay'] = hum_baseline_stay
    merged['human_cog_reward_mod'] = hum_reward_mod
    merged['human_cog_timescale_bal'] = hum_timescale_bal

    # 6. Reorder Columns
    # Structure: Params -> Synaptic -> Network -> Cognitive -> Distances -> Human -> Path
    
    # Identify Params (from Synaptic/Cognitive merge)
    # Prioritize specific order
    param_cols = ['run_id', 'reg_type', 'reg_value', 'l1', 'beta', 'alpha', 'temp', 'travel', 'eps']
    # Removing duplicates if any exists and picking only present
    param_cols = [c for c in param_cols if c in merged.columns]
    
    syn_cols = ['peak_fraction', 'pruning_fraction', 'correlation', 'age_cap', 'peak_age', 'decay_rate']
    net_cols = ['modularity', 'modularity_louvain', 'modularity_leiden',
                'efficiency', 'rich_club', 'small_worldness', 'participation',
                'z_mod', 'z_eff', 'z_rc']
    cog_cols = ['baseline_stay', 'reward_modulation', 'timescale_balance',
                'fitted_alpha_hl', 'fitted_beta_hl', 'beta0', 'beta1', 'std_X']
    
    perf_cols = ['mean_accuracy', 'fraction_solved', 'n_tasks_solved', 'n_tasks_total',
                 'mean_speed', 'switch_rate', 'entropy']

    dist_cols = ['dist_synaptic', 'dist_network', 'dist_cognitive',
                 'norm_dist_synaptic', 'norm_dist_network', 'norm_dist_cognitive',
                 'grand_distance']

    human_cols = [
        'human_syn_peak', 'human_syn_decay', 'human_syn_corr',
        'human_net_modularity', 'human_net_efficiency', 'human_net_rich_club',
        'human_cog_baseline_stay', 'human_cog_reward_mod', 'human_cog_timescale_bal'
    ]
    
    # Construct Path: logs/sweep/forage_v5/run_id
    # We assume standard path
    merged['path'] = "logs/sweep/forage_v5/" + merged['run_id']
    
    # Select columns
    final_cols = param_cols + syn_cols + net_cols + cog_cols + perf_cols + dist_cols + human_cols + ['path']
    
    # Filter only existing
    final_cols = [c for c in final_cols if c in merged.columns]
    
    final_df = merged[final_cols]
    
    # Print Missing Stats
    print("\nMissing Data Summary:")
    print(final_df[['dist_synaptic', 'dist_network', 'dist_cognitive']].isna().sum())
    
    missing_runs = final_df[final_df['grand_distance'].isna()]['run_id'].tolist()
    if missing_runs:
        print(f"\nRuns with missing data ({len(missing_runs)}): {missing_runs}")
    
    # Find winner
    if not final_df['grand_distance'].isna().all():
        winner = final_df.loc[final_df['grand_distance'].idxmin()]
        print(f"\nGrand Unified Winner: {winner['run_id']} (Total Dist: {winner['grand_distance']:.4f})")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV} with {len(final_df)} rows.")

if __name__ == "__main__":
    main()
