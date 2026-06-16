
import numpy as np
import matplotlib.pyplot as plt
from curiosity.data import make_dataset
import os

ALL_KHONA_TASKS = [
    # Family 1: Go
    'khona.go', 'khona.rtgo', 'khona.dlygo', 
    'khona.anti', 'khona.dlyanti', 'khona.rtanti', 
    # Family 2: Decision Making
    'khona.dm1', 'khona.dm2', 'khona.multidm', 
    'khona.ctxdm1', 'khona.ctxdm2', 
    'khona.ctxdlydm1', 'khona.ctxdlydm2', 
    'khona.multidlydm', 
    'khona.dlydm1', 'khona.dlydm2', 
    # Family 3: Memory
    'khona.dms', 'khona.dnms', 'khona.dmc', 'khona.dnmc'
]

POLI_TASKS = [
    # 1. Reach Family
    'poli.go', 'poli.rtgo', 'poli.dlygo', 
    'poli.antigo', 'poli.dlyantigo', 'poli.rtantigo',
    'poli.ctxgo',
    # Level 2 Go
    'poli.dlyctxgo',
    'poli.antictxgo',
    'poli.rtctxgo',
    # Level 3 Go
    'poli.dlyantictxgo',
    'poli.rtantictxgo',
    
    # 2. Decision Family
    'poli.dm1', 'poli.dm2', 'poli.multidm',
    'poli.ctxdm1', 'poli.ctxdm2',
    'poli.dlydm1', 'poli.dlydm2', 'poli.multidlydm',
    'poli.ctxdlydm1', 'poli.ctxdlydm2',
    'poli.antidm1', 'poli.antidm2',
    # Level 2 DM
    'poli.antidlydm1', 'poli.antidlydm2',
    'poli.antictxdm1', 'poli.antictxdm2',
    'poli.antimultidm',
    # Level 3 DM
    'poli.antictxdlydm1', 'poli.antictxdlydm2',
    'poli.antimultidlydm',
    
    # 3. Match Family
    'poli.dlyms', 'poli.dlynms', 
    'poli.catdlyms', 'poli.catdlynms',
    'poli.antidlyms', 'poli.ctxdlyms',
    # Level 2 Match
    'poli.antictxdlyms', 'poli.anticatdlyms',
    'poli.ctxcatdlyms',
    # Level 3 Match
    'poli.antictxcatdlyms',
    # NMS Variants
    'poli.antidlynms', 'poli.ctxdlynms',
    'poli.antictxdlynms', 'poli.anticatdlynms',
    'poli.ctxcatdlynms', 'poli.antictxcatdlynms'
]

# Original Yang19 Tasks (Standard NeuroGym)
# Naming convention verified via find_yang_tasks.py
YANG19_TASKS = [
    'yang19.go-v0', 'yang19.rtgo-v0', 'yang19.dlygo-v0',
    'yang19.anti-v0', 'yang19.dlyanti-v0', 'yang19.rtanti-v0', 
    'yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.multidm-v0',
    'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0',
    'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0',
    'yang19.multidlydm-v0', # Note: 'yang19.dlydm1-v0' etc
    'yang19.dlydm1-v0', 'yang19.dlydm2-v0',
    'yang19.dms-v0', 'yang19.dnms-v0', 'yang19.dmc-v0', 'yang19.dnmc-v0'
]

def visualized_smart_tasks():
    # figures are written into this script's own folder (task_visualizations/)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine lists - Prioritize POLI for now
    tasks_to_run = POLI_TASKS + YANG19_TASKS
    print(f"Visualizing {len(tasks_to_run)} tasks to {output_dir}/ ...")
    
    for task_name in tasks_to_run:
        try:
            print(f"  Generating: {task_name}")
            # Use ds() sampler to enforce sequence length
            # This ensures we capture the full trial (including response) for long tasks like DMS
            ds = make_dataset(task_name, dt=20, batch_size=1, seq_len=1000)
            
            # Generate one batch (Trial)
            inputs, targets = ds()
            
            # Extract first sample in batch
            ob = inputs[:, 0, :]
            gt = targets[:, 0]
            
            time_dim = ob.shape[0]
            input_dim = ob.shape[1]
            
            # --- Active Ring Masking (Time-Dependent) ---
            # Logic: Track which ring is active based on input energy.
            # - If active (E > 0.1), update state.
            # - If silent (Delay), hold previous state.
            
            mask_r1 = np.zeros(time_dim, dtype=bool)
            mask_r2 = np.zeros(time_dim, dtype=bool)
            
            # Initial state: Default to Ring 1 (standard)
            state_r1 = True
            state_r2 = False
            
            has_r2 = (input_dim >= 33)
            
            # Threshold: 0.1 to avoid low-level noise triggering false positives
            E_THRESH = 0.1
            
            for t in range(time_dim):
                # Check instantaneous energy
                e1 = np.mean(ob[t, 1:17])
                e2 = np.mean(ob[t, 17:33]) if has_r2 else 0.0
                
                is_on_1 = e1 > E_THRESH
                is_on_2 = e2 > E_THRESH
                
                # State Update Logic
                if is_on_1 and is_on_2:
                    state_r1 = True
                    state_r2 = True
                elif is_on_1:
                    state_r1 = True
                    state_r2 = False
                elif is_on_2:
                    state_r1 = False
                    state_r2 = True
                # Else (Silence): Keep previous state
                
                mask_r1[t] = state_r1
                mask_r2[t] = state_r2

            # Setup Single Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Background: Sensory Input
            ax.imshow(ob.T, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis', alpha=0.9)
            ax.set_title(f"Task: {task_name}", fontsize=14)
            ax.set_ylabel("Sensory Inputs + Target (White)\n(0=Fix, 1-16=Ring1, 17-32=Ring2)", color='black', fontsize=10)
            
            # --- Heatmap Style Target Overlay ---
            # Create an RGBA image for the target
            target_rgba = np.zeros((input_dim, time_dim, 4)) # RGBA
            
            # Fill Target Pixels based on Hybrid Logic with Forced Overrides
            
            # --- Strict Task Overrides ---
            # Some tasks explicitly define which ring is relevant.
            # We strictly enforce this to avoid "ghost" targets on distractor rings.
            force_ring = None
            force_both = False

            if 'multi' in task_name:
                force_both = True # Always plot on both rings for MultiDM
            elif 'dm2' in task_name or 'catdm2' in task_name:
                force_ring = 16 # Ring 2 (Shift)
            elif 'dm1' in task_name or 'catdm1' in task_name:
                force_ring = 0 # Ring 1
            elif 'ctx' in task_name or 'context' in task_name:
                 # Context implies attend Mod 1 (since Mod 2 is distractor)
                 # Unless it was already caught by dm2 above
                 force_ring = 0


            for t in range(time_dim):
                cls = gt[t]
                if cls > 0:
                    idx0 = int(cls)
                    idx1 = int(cls + 16)
                    
                    # If Multi-Sensory, force both rings
                    if force_both:
                         if 0 <= idx0 < input_dim:
                             target_rgba[idx0, t, :] = [1.0, 1.0, 1.0, 1.0]
                         if has_r2 and 0 <= idx1 < input_dim:
                             target_rgba[idx1, t, :] = [1.0, 1.0, 1.0, 1.0]
                         continue

                    # If we have a forced ring, just plot there and skip logic
                    if force_ring is not None:
                         idx_forced = int(cls + force_ring)
                         if 0 <= idx_forced < input_dim:
                             target_rgba[idx_forced, t, :] = [1.0, 1.0, 1.0, 1.0]
                         continue

                    # --- Default Hybrid Logic (for Anti, Go, etc.) ---
                    
                    # Local intensity check
                    val0 = ob[t, idx0] if idx0 < input_dim else 0.0
                    val1 = ob[t, idx1] if (has_r2 and idx1 < input_dim) else 0.0
                    
                    has_signal0 = val0 > E_THRESH
                    has_signal1 = val1 > E_THRESH
                    
                    if has_signal0 or has_signal1:
                        # Signal present at target! Use local info to be precise.
                        if has_signal0:
                            target_rgba[idx0, t, :] = [1.0, 1.0, 1.0, 1.0]
                        if has_signal1:
                            target_rgba[idx1, t, :] = [1.0, 1.0, 1.0, 1.0]
                                
                    else:
                        # No signal at target location (Anti, Delay, or Silence).
                        # Fall back to Active Mask
                        if mask_r1[t]:
                            if 0 <= idx0 < input_dim:
                                target_rgba[idx0, t, :] = [1.0, 1.0, 1.0, 1.0]
                        if mask_r2[t] and has_r2:
                            if 0 <= idx1 < input_dim:
                                target_rgba[idx1, t, :] = [1.0, 1.0, 1.0, 1.0]
            
            # Overlay Target Heatmap
            ax.imshow(target_rgba, aspect='auto', interpolation='nearest', origin='lower')
            
            # --- Dual Y-Ticks (Repeated 1-16) ---
            # Range 1: 1-16 (Ring 1)
            # Range 2: 17-32 (Ring 2)
            ticks_r1 = np.arange(1, 17)
            ticks_r2 = np.arange(17, 33)
            
            all_ticks = np.concatenate([ticks_r1, ticks_r2])
            all_labels = list(range(1, 17)) + list(range(1, 17))
            all_labels = [str(l) for l in all_labels]
            
            # Filter ticks within view
            valid_mask = all_ticks < input_dim
            final_ticks = all_ticks[valid_mask]
            final_labels = np.array(all_labels)[valid_mask]
            
            # We use the primary axis for everything now since we overlay directly
            ax.set_yticks(final_ticks)
            ax.set_yticklabels(final_labels, fontsize=8)
            
            # Divider Lines
            ax.axhline(y=0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axhline(y=16.5, color='black', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel("Time (steps)")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{task_name}.png")
            plt.close()
            
        except Exception as e:
            print(f"  FAILED {task_name}: {e}")

    print("Done.")

if __name__ == "__main__":
    visualized_smart_tasks()
