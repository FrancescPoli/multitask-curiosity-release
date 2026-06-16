"""
Extract L1 sum and L0 norm of W_in, W_rec, W_out for every run in a sweep.

Implements the diagnostic check in future_steps.md section 4.1: do unregularized
input/readout weights inflate in the Distance branch relative to the L1 branch?

Mirrors the metric definitions used for W_rec in
analysis/extract_weights_evolution.py (l1_sum, l0_norm with threshold 1e-5).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from curiosity.utils.model_loader import (
    build_model_from_meta,
    load_meta,
    load_state_into_model,
)


L0_THRESH = 1e-5


def get_Wrec(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "cell_sep") and getattr(model, "cell_sep") is not None:
        return model.cell_sep.Wrec
    if hasattr(model, "cell") and getattr(model, "cell") is not None:
        n_in = int(model.hp.n_input)
        return model.cell.W[n_in:, :]
    raise RuntimeError("Could not locate W_rec.")


def get_Win(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "cell_sep") and getattr(model, "cell_sep") is not None:
        sensory = model.sensory_in.weight
        rule = model.rule_in.weight
        return torch.cat([sensory, rule], dim=1)
    if hasattr(model, "cell") and getattr(model, "cell") is not None:
        n_in = int(model.hp.n_input)
        return model.cell.W[:n_in, :]
    raise RuntimeError("Could not locate W_in.")


def get_Wout(model: nn.Module) -> torch.Tensor:
    return model.w_out.weight


def norms(W: torch.Tensor) -> Tuple[float, int]:
    w_abs = W.detach().abs()
    return float(w_abs.sum().item()), int((w_abs > L0_THRESH).sum().item())


def compute_run_norms(run_dir: Path, device: torch.device) -> dict:
    meta = load_meta(run_dir)
    sd_path = run_dir / "state_dict.pt"
    if not sd_path.exists():
        raise FileNotFoundError(f"No state_dict.pt in {run_dir}")

    state = torch.load(sd_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = build_model_from_meta(meta, device=device)
    load_state_into_model(model, state)
    model.eval()

    w_in_l1, w_in_l0 = norms(get_Win(model))
    w_rec_l1, w_rec_l0 = norms(get_Wrec(model))
    w_out_l1, w_out_l0 = norms(get_Wout(model))

    return {
        "run_id": run_dir.name,
        "w_in_l1_sum": w_in_l1,
        "w_in_l0": w_in_l0,
        "w_rec_l1_sum": w_rec_l1,
        "w_rec_l0": w_rec_l0,
        "w_out_l1_sum": w_out_l1,
        "w_out_l0": w_out_l0,
    }


def compute_init_norms(sweep_dir: Path, device: torch.device) -> Optional[dict]:
    """Init weights are identical across runs (seed=42 + deterministic init).
    Compute from the first run that has state_step000000.pt available."""
    for run_dir in sorted(sweep_dir.iterdir()):
        if not (run_dir.is_dir() and run_dir.name.startswith("run_")):
            continue
        init_ckpt = run_dir / "state_step000000.pt"
        meta_path = run_dir / "model_meta.json"
        if not (init_ckpt.exists() and meta_path.exists()):
            continue

        meta = load_meta(run_dir)
        obj = torch.load(init_ckpt, map_location=device)
        state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        model = build_model_from_meta(meta, device=device)
        load_state_into_model(model, state)
        model.eval()

        w_in_l1, w_in_l0 = norms(get_Win(model))
        w_rec_l1, w_rec_l0 = norms(get_Wrec(model))
        w_out_l1, w_out_l0 = norms(get_Wout(model))
        return {
            "source_run": run_dir.name,
            "w_in_l1_sum": w_in_l1,
            "w_in_l0": w_in_l0,
            "w_rec_l1_sum": w_rec_l1,
            "w_rec_l0": w_rec_l0,
            "w_out_l1_sum": w_out_l1,
            "w_out_l0": w_out_l0,
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir",
        type=str,
        default="Z:/fp02/logs/sweep/forage_v5.1/forage_v6",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/Regularization_scope/io_weight_norms.csv",
    )
    parser.add_argument(
        "--init_output",
        type=str,
        default="analysis/Regularization_scope/io_weight_norms_init.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.output)
    init_path = Path(args.init_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not sweep_dir.exists():
        print(f"Sweep dir not found: {sweep_dir}")
        sys.exit(1)

    device = torch.device("cpu")

    # Init norms (one-time, since seed=42 is fixed across all runs)
    if not init_path.exists():
        print("Computing init norms (one-time)...")
        init = compute_init_norms(sweep_dir, device)
        if init is None:
            print("WARNING: no state_step000000.pt found in any run; skipping init.")
        else:
            pd.DataFrame([init]).to_csv(init_path, index=False)
            print(f"  Init: {init}")
            print(f"  Saved -> {init_path}")
    else:
        print(f"Init already cached at {init_path}")

    # Resume
    processed = set()
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            if "run_id" in existing.columns:
                processed = set(existing["run_id"].unique())
                print(f"Resuming: {len(processed)} runs already processed.")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    runs = sorted(
        d for d in sweep_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if args.limit:
        runs = runs[: args.limit]

    print(f"Total runs to consider: {len(runs)}")

    added, skipped, failed = 0, 0, 0
    for i, run_dir in enumerate(runs, 1):
        if run_dir.name in processed:
            skipped += 1
            continue
        try:
            row = compute_run_norms(run_dir, device)
        except FileNotFoundError as e:
            print(f"[{i}/{len(runs)}] SKIP {run_dir.name}: {e}")
            failed += 1
            continue
        except Exception as e:
            print(f"[{i}/{len(runs)}] FAIL {run_dir.name}: {e}")
            failed += 1
            continue

        pd.DataFrame([row]).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False
        )
        processed.add(run_dir.name)
        added += 1
        if i % 50 == 0 or i == len(runs):
            print(f"[{i}/{len(runs)}] added={added} skipped={skipped} failed={failed}")

    print(f"\nDone. added={added} skipped={skipped} failed={failed}")
    print(f"Output -> {out_path}")


if __name__ == "__main__":
    main()
