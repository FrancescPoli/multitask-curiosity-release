"""Per-run, per-TASK (all training tasks) curriculum stats for the Goldilocks
analysis. One pass per run_curriculum.csv reading only the 4 needed columns.

Resilient to transient Z: network drops (WinError 64): each run is retried a
few times, results are APPENDED to the CSV after every run (so a crash loses
nothing), and on restart already-processed runs are skipped (resume).

Per task the network trained:
  - samp_frac      : fraction of that network's TRAIN steps spent on the task
  - final_acc      : end-of-training accuracy (last train-row EMA)
  - steps_to_crit  : first train_step at which EMA >= CRIT (NaN if never)
  - total_train_steps
Run via PowerShell (Z: visible)."""
from pathlib import Path
import os
import time
import pandas as pd

SWEEP = Path("Z:/fp02/logs/sweep/forage_v5.1/forage_v6")
CRIT = 0.8
OUT = "analysis/Cognitive_analysis/Goldilocks/tmp_alltask_out.csv"
MOUNT_POLL = 20         # seconds between mount-reachability checks
MOUNT_MAX_WAIT = 1800   # stop (for later resume) if mount stays down this long

def per_run(cp):
    df = pd.read_csv(cp, usecols=["task", "mode", "ema", "train_step"],
                     on_bad_lines="skip")
    tr = df[df["mode"] == "train"]
    n = len(tr)
    if n == 0:
        return None
    samp  = tr["task"].value_counts() / n
    final = tr.dropna(subset=["ema"]).groupby("task", sort=False)["ema"].last()
    crit  = tr[tr["ema"] >= CRIT].groupby("task")["train_step"].min()
    out = pd.DataFrame({"samp_frac": samp}) \
            .join(final.rename("final_acc")) \
            .join(crit.rename("steps_to_crit"))
    out["total_train_steps"] = n
    out = out.reset_index().rename(columns={"index": "task"})
    return out

def mount_up():
    try:
        return SWEEP.exists()
    except OSError:
        return False

def wait_for_mount():
    """Block until the Z: mount root is reachable again. False if it stays down."""
    waited = 0
    while not mount_up():
        if waited >= MOUNT_MAX_WAIT:
            return False
        time.sleep(MOUNT_POLL); waited += MOUNT_POLL
        print(f"    mount down; waited {waited}s", flush=True)
    return True

def process_run(run_id):
    """Read one run, WAITING OUT transient mount drops rather than skipping them.
    Returns a DataFrame, None (file genuinely absent), or 'MOUNT_DOWN'."""
    cp = SWEEP / run_id / "run_curriculum.csv"
    while True:
        try:
            if not cp.exists():
                if mount_up():
                    return None                # mount up, file genuinely absent
                if not wait_for_mount():
                    return "MOUNT_DOWN"
                continue                       # mount back; re-check the file
            return per_run(cp)
        except OSError as e:
            print(f"    network error ({e}); waiting for mount...", flush=True)
            if not wait_for_mount():
                return "MOUNT_DOWN"
            # mount back; loop and retry the read

runs = pd.read_csv("analysis/compositional_analysis/Data/compositional_dataset.csv")[["run_id", "cohort"]].drop_duplicates()
print(f"{len(runs)} cohort runs", flush=True)

done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, usecols=["run_id"])["run_id"].unique())
    print(f"resuming: {len(done)} runs already done", flush=True)
write_header = not os.path.exists(OUT)

for i, r in enumerate(runs.itertuples(index=False)):
    if r.run_id in done:
        continue
    try:
        o = process_run(r.run_id)
    except Exception as e:
        print(f"  ERR {r.run_id}: {e}", flush=True); continue
    if isinstance(o, str) and o == "MOUNT_DOWN":
        print(f"  MOUNT DOWN > {MOUNT_MAX_WAIT}s at {r.run_id}; "
              f"stopping — re-run to resume.", flush=True)
        break
    if o is None:
        continue
    o.insert(0, "cohort", r.cohort)
    o.insert(0, "run_id", r.run_id)
    o.to_csv(OUT, mode="a", header=write_header, index=False)
    write_header = False
    print(f"  [{i+1}/{len(runs)}] {r.run_id} ({len(o)} tasks)", flush=True)

allt = pd.read_csv(OUT)
print(f"\nTotal {len(allt)} task-rows for {allt['run_id'].nunique()} runs", flush=True)
print("tasks seen:", allt["task"].nunique(), flush=True)
print("DONE", flush=True)
