from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

# --- Supervised loss over all non-padding frames (robust to -1 or -100)
def ce_all_nonpadding_frames(logits: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over all frames with y >= 0.
    logits: (T,B,C), y_labels: (T,B) where negatives are padding/ignore.
    """
    T, B, C = logits.shape
    logits2d = logits.view(-1, C)
    y = y_labels.view(-1).long().to(logits.device)
    valid = (y >= 0)
    if not valid.any():
        # return a correct-device scalar zero
        return logits2d.sum() * 0.0
    return F.cross_entropy(logits2d[valid], y[valid], reduction='mean')


def ce_yang_cmask(logits: torch.Tensor, y_labels: torch.Tensor, decision_weight: float = 5.0) -> torch.Tensor:
    """
    Cross-entropy with Yang-style c_mask temporal weighting.
    
    Matches Yang et al. (2019) original implementation where:
    - Decision frames (y > 0): weighted 5.0× 
    - Fixation frames (y == 0): weighted 1.0×
    - Invalid frames (y < 0): ignored
    
    This prevents gradient dilution from long fixation/delay periods.
    
    Args:
        logits: (T,B,C) - model outputs
        y_labels: (T,B) - target labels (0=fixation, >0=decision direction, <0=padding)
        decision_weight: weight multiplier for decision frames (default 5.0, Yang's value)
    
    Returns:
        Weighted cross-entropy loss (scalar)
    """
    T, B, C = logits.shape
    logits2d = logits.view(-1, C)
    y = y_labels.view(-1).long().to(logits.device)
    
    # Create weights: decision frames get higher weight
    decision_mask = (y > 0).float()
    fixation_mask = (y == 0).float()
    valid_mask = (y >= 0)
    
    if not valid_mask.any():
        return logits2d.sum() * 0.0
    
    # Temporal importance weights (matching Yang's c_mask)
    weights = decision_mask * decision_weight + fixation_mask * 1.0
    
    # Compute per-frame loss
    loss_per_frame = F.cross_entropy(logits2d, y, reduction='none')
    
    # Apply weights and mask invalid frames
    weighted_loss = loss_per_frame * weights
    masked_loss = weighted_loss[valid_mask]
    
    return masked_loss.mean()



# --- Optional hidden-activity regularization (ℓ1/ℓ2 on h(t))
def activity_reg(h_seq: torch.Tensor, l1_h: float = 0.0, l2_h: float = 0.0) -> torch.Tensor:
    """
    h_seq: (T,B,H). Returns 0 if both weights are 0.
    """
    reg = h_seq.sum() * 0.0  # scalar zero on the right device/dtype
    if l1_h > 0:
        reg = reg + l1_h * h_seq.abs().mean()
    if l2_h > 0:
        reg = reg + 0.5 * l2_h * (h_seq.pow(2).mean())
    return reg

def batch_decision_accuracy_seqwise(logits2d: torch.Tensor, y_flat: torch.Tensor, T: int, B: int) -> float:
    C = logits2d.size(-1)
    logits_TBC = logits2d.reshape(T, B, C)
    acc = 0; total = 0
    with torch.no_grad():
        y_TB = y_flat.view(T, B)
        for b in range(B):
            gt = y_TB[:, b].detach().cpu().numpy()
            valid = np.where(gt > 0)[0]
            if valid.size == 0:
                continue
            t = int(valid[-1])
            pred = logits_TBC[t, b].argmax(dim=-1).item()
            acc += int(pred == int(gt[t]))
            total += 1
    return (acc / total) if total > 0 else 0.0

def ce_loss_on_decision_frames(ring_logits: torch.Tensor, y_labels: torch.Tensor, n_actions: int):
    """Cross-entropy averaged only over valid decision frames (y ∉ {-1,0}).
    Returns (loss or None if no valid frames, logits2d, valid_mask_2d)."""
    T, B, _ = ring_logits.shape
    valid = (y_labels > 0)
    logits2d = ring_logits.view(-1, n_actions)
    if not valid.any():
        return None, logits2d, valid.view(-1).float()
    idx2d = valid.view(-1)
    targets = y_labels[valid]
    loss = F.cross_entropy(logits2d[idx2d], targets, reduction='mean')
    return loss, logits2d, idx2d.float()

def evaluate_task(model, env, device='cpu', ntrial: int = 200, rule_idx: int = None, n_rules: int = None, train_mode: bool = False, sigma_x: float = 0.0) -> float:
    """Evaluates accuracy: take the last valid decision frame in each trial and compare argmax to label.
    Compatible with Yang19KhonaModel (expects forward() to return a dict with 'ring_logits')."""
    was_training = model.training
    if train_mode:
        model.train() # Force training mode (with noise)
    else:
        model.eval()
        
    correct = 0; total = 0
    with torch.no_grad():
        # print("DEBUG: Starting eval loop", flush=True)
        for _ in range(ntrial):
            # print(f"DEBUG: Loop iter {_}", flush=True)
            # API compatibility: try new_trial(), fall back if needed
            if hasattr(env, 'new_trial'):
                _ = env.new_trial()
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'new_trial'):
                _ = env.unwrapped.new_trial()
            else:
                 pass

            # Access observation
            if hasattr(env, 'ob'):
                ob = env.ob
                gt = env.gt
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'ob'):
                ob = env.unwrapped.ob
                gt = env.unwrapped.gt
            else:
                # print("DEBUG: No ob found, continuing", flush=True)
                continue
            
            # print(f"DEBUG: Got ob shape {ob.shape}", flush=True)

            # Manually append rule inputs if requested
            if rule_idx is not None and n_rules is not None:
                T = ob.shape[0] if len(ob.shape) > 1 else 1 # Safety check
                if len(ob.shape) == 1: ob = ob[None, :] # Make (1, Dim) if needed? No, NeuroGym ob is (T, D) usually.
                
                # Create (Seq, N_RULES)
                # Ensure ob is (T, D)
                if len(ob.shape) == 1:
                     ob = ob[None, :]
                     
                T, _ = ob.shape 
                rule_hot = np.zeros((T, n_rules), dtype=ob.dtype)
                rule_hot[:, rule_idx] = 1.0
                ob = np.concatenate([ob, rule_hot], axis=-1)

            x = torch.from_numpy(ob[:, None, :]).float().to(device)
            
            # Inject noise (Yang-style)
            # sigma_x is effectively scaled by sqrt(2/alpha). Assuming alpha=0.2 (dt=20/tau=100)
            if sigma_x > 0:
                alpha = 0.2
                noise_std = sigma_x * (2.0 / alpha)**0.5
                x = x + torch.randn_like(x) * noise_std
            
            # DEBUG: Print input stats once
            if _ == 0:
                print(f"DEBUG INPUT: x.shape={x.shape}", flush=True)
                
                nz = torch.nonzero(x[0, 0, :], as_tuple=False)
                print(f"DEBUG: Non-zero indices at t=0: {nz.flatten().tolist()}", flush=True)
                
                start_rule = x.shape[-1] - n_rules
                print(f"DEBUG: Rule vector (last {n_rules} dims) at t=0: {x[0,0,start_rule:].tolist()}", flush=True)
            
            outs = model(x)
            logits = outs["ring_logits"]
            valid = np.where((gt != -1) & (gt != 0))[0]
            if valid.size == 0:
                continue
            t = int(valid[-1])
            pred = logits[t, 0].argmax().item()
            correct += int(pred == int(gt[t]))
            total += 1
            
    # Restore original state if we changed it, or leave as is?
    # Safer to restore.
    model.train(was_training)
    return (correct / total) if total > 0 else 0.0
