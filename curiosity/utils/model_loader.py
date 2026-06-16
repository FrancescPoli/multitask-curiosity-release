"""
Centralized model loading utilities for Yang19KhonaModel.
Refactored from load_trained_model.py and compute_dynamics.py.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import json
import torch
import curiosity.models.yang19_khona as tyk

def load_meta(run_dir: Path) -> Dict[str, Any]:
    """Load model_meta.json from a run directory."""
    meta_path = run_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find model_meta.json in {run_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_state_dict_from_run(
    run_dir: Path,
    map_location: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load a raw state_dict (name -> tensor) from state_dict.pt or model_last.pt.
    
    - Prefers state_dict.pt (the pure state_dict).
    - Falls back to model_last.pt (checkpoint; may contain 'state_dict' key).
    """
    sd_path = run_dir / "state_dict.pt"
    ckpt_path = run_dir / "model_last.pt"

    if sd_path.exists():
        path = sd_path
    elif ckpt_path.exists():
        path = ckpt_path
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir} (looked for state_dict.pt and model_last.pt)."
        )

    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint at {path} did not contain a valid state_dict.")

    return state

def build_model_from_meta(
    meta: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Recreate Yang19KhonaModel using the saved HParams + kwargs in meta.
    """
    hp_dict = meta.get("hparams", {})
    hp = tyk.HParams(**hp_dict)

    model_class_name = meta.get("model_class", "Yang19KhonaModel")
    ModelCls = getattr(tyk, model_class_name, tyk.Yang19KhonaModel)
    model_kwargs = meta.get("model_kwargs", {})

    model = ModelCls(hp, **model_kwargs).to(device)
    return model

def load_state_into_model(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """
    Load a state_dict into model, registering optional buffers if needed.
    """
    # Register any optional buffers present in state but missing on model
    for optbuf in ("brain_mask", "brain_dale_signs", "distance_matrix"):
        if optbuf in state and optbuf not in model.state_dict():
            model.register_buffer(optbuf, state[optbuf])

    model.load_state_dict(state, strict=True)
