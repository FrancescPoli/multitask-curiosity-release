from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json

def _try_dataclass_to_dict(obj) -> Dict[str, Any]:
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    return None

def _to_jsonable(obj):
    # Fast path for builtins
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy scalars/arrays
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Torch types
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            # avoid dumping huge arrays; store summary
            if obj.numel() == 1:
                return _to_jsonable(obj.item())
            return {"__torch_tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, (torch.device, torch.dtype)):
            return str(obj)
    except Exception:
        pass

    # Dataclass?
    maybe_dc = _try_dataclass_to_dict(obj)
    if maybe_dc is not None:
        return _to_jsonable(maybe_dc)

    # Mappings
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # Iterables
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # Path-like
    if hasattr(obj, "__fspath__"):
        try:
            return str(obj)
        except Exception:
            pass

    # Generic objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(dict(obj.__dict__))
        except Exception:
            pass

    # Fallback to string
    return str(obj)

def save_checkpoint(model, path: Union[str, Path], meta: Optional[Dict[str, Any]] = None) -> Path:
    """Saves a torch checkpoint with state_dict + optional metadata."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta or {}
    }
    torch.save(payload, path)
    return path

def save_state_dict(model, path: Union[str, Path]) -> Path:
    """Saves only the raw state_dict (simplest format)."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path

def load_state_into_model(model, path: Union[str, Path], map_location: str = "cpu"):
    """Loads weights from either a raw state_dict file or a checkpoint payload."""
    import torch
    path = Path(path)
    obj = torch.load(path, map_location=map_location)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)
    return model

def write_json(obj: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, sort_keys=True)
    return path
