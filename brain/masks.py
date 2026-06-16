
from __future__ import annotations
import os
import numpy as np
import numpy.linalg as npl
import torch

def symmetrize_zero_diag(A: np.ndarray) -> np.ndarray:
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A

def normalize_to_spectral_radius(W: np.ndarray, target_sr: float = 0.9) -> np.ndarray:
    if not np.any(W):
        return W.astype(np.float32)
    vals = npl.eigvals(W)
    sr = float(np.max(np.abs(vals)))
    if sr > 0:
        W = (target_sr / sr) * W
    return W.astype(np.float32)

def _load_array(path_or_array):
    if isinstance(path_or_array, str):
        ext = os.path.splitext(path_or_array)[1].lower()
        if ext in ('.pt', '.pth'):
            arr = torch.load(path_or_array, map_location='cpu')
            if isinstance(arr, dict) and 'A' in arr:
                arr = arr['A']
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
        else:
            arr = np.load(path_or_array)
    else:
        arr = path_or_array
    return np.asarray(arr, dtype=np.float32)

def load_connectome(path_or_array, *, target_sr: float = 0.9):
    A = _load_array(path_or_array)
    A = symmetrize_zero_diag(A)
    mask = (A != 0).astype(np.float32)
    A_scaled = normalize_to_spectral_radius(A, target_sr=target_sr)
    return mask, A_scaled
