#!/usr/bin/env python3
"""
Cluster Submission Manager.
PURE EXECUTION ONLY. No model configuration logic here.

Usage:
    # 1. Basic Mode (Generate Array Scripts with Node Selection)
    python experiments/run_sweep_slurm.py --sweep-id my_sweep --mode basic
    
    # 2. Daemon Mode (Active Management)
    python experiments/run_sweep_slurm.py --sweep-id my_sweep --mode daemon
    
    # 3. Vanilla Mode (Simple Submission, No Node Logic)
    python experiments/run_sweep_slurm.py --sweep-id my_sweep --mode vanilla
"""

import argparse
import sys
import itertools
import subprocess
import time
import getpass
import datetime
from pathlib import Path

# Import CONFIGS from the main sweep script (The Source of Truth)
sys.path.append(str(Path(__file__).parent.parent))
from experiments.run_sweep import CONFIGS as REG_CONFIGS
from experiments.run_sweep_forage import CONFIGS as FORAGE_CONFIGS
from analysis.utils.paths import get_logs_dir

CONFIGS = {**REG_CONFIGS, **FORAGE_CONFIGS}

# Known problematic nodes - always excluded regardless of mode
# Update this list as you discover hardware issues
KNOWN_BAD_NODES = ['node-k01', 'node-k12']#['node-k01', 'node-k12']

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Sweep Manager")
    parser.add_argument("--sweep-id", type=str, required=True, help="Name of the sweep folder")
    parser.add_argument("--config", type=str, default="default", choices=["default", "yang_og", "forage", "forage_v2", "forage_v3", "forage_v4", "forage_v5", "forage_v6", "forage_v7", "forage_v8"], help="Choose sweep configuration")
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "daemon", "vanilla"], 
                        help="Operation mode: 'basic' generates sbatch scripts with node selection, 'daemon' actively manages queue, 'vanilla' generates simple scripts without any node logic.")
    parser.add_argument("--strategy", type=str, default="best", choices=["exclude", "best"], 
                        help="Node strategy: 'exclude' bans bad nodes, 'best' targets healthy nodes JIT.")
    parser.add_argument("--log-root", type=str, default="logs/sweep", help="Root logging directory")
    parser.add_argument("--base-dir", type=str, default=None, help="Override output directory directly (skip log-root/sweep-id/config nesting). Use to add V7 runs into an existing V6 folder.")
    # Cluster constraints
    parser.add_argument("--partition", type=str, default="Main", help="Slurm partition")
    parser.add_argument("--queue", type=str, default="normal", help="Slurm QOS (normal|lopri)")
    parser.add_argument("--time", type=str, default="48:00:00", help="Time limit")
    parser.add_argument("--mem", type=str, default="8G", help="Memory per CPU")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per task")
    args = parser.parse_args()
    if args.log_root == "logs/sweep":
        args.log_root = str(get_logs_dir() / "sweep")
    return args

def scan_nodes(required_cpus=1):
    """
    Scans cluster for Best and Bad nodes.
    Returns: (list_of_bad, list_of_best)
    """
    bad_nodes = [] 
    best_nodes = []
    
    try:
        cmd = ["sinfo", "-N", "-h", "-o", "%N|%O|%C"]
        output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        
        for line in output.split('\n'):
            if not line.strip(): continue
            try:
                parts = line.split('|')
                name = parts[0]
                load = float(parts[1]) if parts[1] != 'N/A' else 0.0
                state_parts = parts[2].split('/')
                idle_cpus = int(state_parts[1])
                total_cpus = int(state_parts[3])
                
                if load > (total_cpus * 1.5):
                    if name not in bad_nodes: bad_nodes.append(name)
                
                if idle_cpus >= required_cpus and load < (total_cpus * 0.8):
                    if name not in bad_nodes: best_nodes.append(name)
            except: continue
    except Exception as e:
        print(f"[Warn] Node scan failed: {e}")
        
    # Always include known bad nodes
    bad_nodes.extend(KNOWN_BAD_NODES)
    return sorted(list(set(bad_nodes))), sorted(list(set(best_nodes)))

def get_queue_status(user):
    status = { "normal_pd": 0, "normal_r": 0, "lopri_pd": 0, "lopri_r": 0 }
    try:
        cmd = ["squeue", "-u", user, "-h", "-o", "%q|%t"]
        output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        for line in output.split('\n'):
            if not line.strip(): continue
            parts = line.split('|')
            if len(parts) < 2: continue
            qos = parts[0].lower()
            state = parts[1]
            if "normal" in qos:
                if state == "PD": status["normal_pd"] += 1
                elif state == "R": status["normal_r"] += 1
            elif "lopri" in qos:
                if state == "PD": status["lopri_pd"] += 1
                elif state == "R": status["lopri_r"] += 1
    except: pass
    return status

def generate_script_content(run_name, cmd_str, args, base_dir, qos, node_line, python_script="run.py"):
    """Generates the full bash script content for a single job."""
    return f"""#!/bin/bash
#SBATCH -J {run_name}
#SBATCH -o {base_dir.as_posix()}/{run_name}.out
#SBATCH -e {base_dir.as_posix()}/{run_name}.err
#SBATCH -p {args.partition}
#SBATCH -q {qos}
#SBATCH -t {args.time}
#SBATCH -N 1
#SBATCH -c {args.cpus}
#SBATCH --mem-per-cpu={args.mem}
{node_line}

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"
conda activate ngym39

export OMP_NUM_THREADS={args.cpus}
export MKL_NUM_THREADS={args.cpus}
export NUMEXPR_NUM_THREADS={args.cpus}
export TORCH_NUM_THREADS={args.cpus}

if [ -f "/imaging/astle/fp02/RNN/{python_script}" ]; then cd "/imaging/astle/fp02/RNN";
elif [ -d "multitask-curiosity" ]; then cd multitask-curiosity; fi

python -u {python_script} {cmd_str}
"""

def submit_job_daemon(run_name, cmd_str, args, base_dir, qos, node_line, python_script="run.py"):
    script = generate_script_content(run_name, cmd_str, args, base_dir, qos, node_line, python_script)
    try:
        res = subprocess.run(["sbatch"], input=script, universal_newlines=True, capture_output=True, check=True)
        return True, res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def run_basic_mode(args, job_list, base_dir, bad_nodes, best_nodes):
    print(">>> MODE: BASIC (Generating Array Script)")
    slurm_node_line = ""
    if args.strategy == "best":
        if best_nodes:
            print(f"Targeting {len(best_nodes)} healthy nodes.")
            slurm_node_line = f"#SBATCH --nodelist={','.join(best_nodes)}"
        else:
            print("No ideal nodes found, switching to exclude.")
            if bad_nodes: slurm_node_line = f"#SBATCH --exclude={','.join(bad_nodes)}"
    elif args.strategy == "exclude":
        if bad_nodes: 
            print(f"Excluding {len(bad_nodes)} overloaded nodes.")
            slurm_node_line = f"#SBATCH --exclude={','.join(bad_nodes)}"

    config_list_path = base_dir / "config_list.txt"
    with open(config_list_path, "w") as f:
        for _, cmd_str in job_list:
            f.write(cmd_str + "\n")
            
    limit_normal = 100
    total = len(job_list)
    scripts = []
    
    def write_array(start, end, qos, suffix):
        if start > end: return
        fname = f"submit_{suffix}.sbatch"
        script = f"""#!/bin/bash
#SBATCH -J {args.sweep_id}_{suffix}
#SBATCH -o {base_dir.as_posix()}/%x_%A_%a.out
#SBATCH -e {base_dir.as_posix()}/%x_%A_%a.err
#SBATCH -p {args.partition}
#SBATCH -q {qos}
#SBATCH -t {args.time}
#SBATCH -N 1
#SBATCH -c {args.cpus}
#SBATCH --mem-per-cpu={args.mem}
{slurm_node_line}
#SBATCH --array={start}-{end}

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"
conda activate ngym39

export OMP_NUM_THREADS={args.cpus}
export MKL_NUM_THREADS={args.cpus}
export NUMEXPR_NUM_THREADS={args.cpus}
export TORCH_NUM_THREADS={args.cpus}

if [ -f "/imaging/astle/fp02/RNN/run.py" ]; then cd "/imaging/astle/fp02/RNN";
elif [ -d "multitask-curiosity" ]; then cd multitask-curiosity; fi

CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {config_list_path.as_posix()})
python -u run.py $CMD
"""
        with open(fname, "w") as f: f.write(script)
        scripts.append(fname)

    if total > limit_normal and args.queue == "normal":
        write_array(0, limit_normal-1, "normal", "prio")
        write_array(limit_normal, total-1, "lopri", "overflow")
    else:
        write_array(0, total-1, args.queue, "all")
        
    print("\nGenerated Scripts:")
    for s in scripts: print(f"  sbatch {s}")

def run_daemon_mode(args, job_list, base_dir, user):
    print(f">>> MODE: DAEMON (Active Management for {len(job_list)} Jobs)")
    job_idx = 0
    while job_idx < len(job_list):
        run_name, cmd_str = job_list[job_idx]
        if (base_dir / run_name / "model_last.pt").exists():
            print(f"[SKIP] Completed: {run_name}")
            job_idx += 1
            continue
            
        status = get_queue_status(user)
        target_qos = None
        if status["normal_pd"] == 0: target_qos = "normal"
        elif status["lopri_pd"] == 0: target_qos = "lopri"
        
        if target_qos is None:
            print(f"\r[WAIT] Queues Busy (Normal: {status['normal_pd']} PD, Lopri: {status['lopri_pd']} PD). Sleeping 10m...", end="", flush=True)
            time.sleep(600)
            continue
            
        bads, bests = scan_nodes(args.cpus)
        node_line = ""
        if args.strategy == "best":
            if bests: node_line = f"#SBATCH --nodelist={','.join(bests)}"
            elif bads: node_line = f"#SBATCH --exclude={','.join(bads)}"
        elif args.strategy == "exclude":
            if bads: node_line = f"#SBATCH --exclude={','.join(bads)}"
            
        success, msg = submit_job_daemon(run_name, cmd_str, args, base_dir, target_qos, node_line)
        if success:
            print(f"\n[SUBMIT] {run_name} -> {target_qos.upper()} ({msg}) on {node_line if node_line else 'ANY'}")
            job_idx += 1
            time.sleep(5)
        else:
            print(f"\n[ERROR] Submit failed: {msg}. Retrying in 10s...")
            time.sleep(10)
    print(">>> All jobs submitted.")

def main():
    args = parse_args()
    
    # 1. Prepare Configuration
    # Select Config using the CLI argument
    selected_config = CONFIGS[args.config]
    base_conf = selected_config["base_args"]
    sets = selected_config["grid"]
    
    # timestamp / sweep-id logic
    if args.sweep_id:
        timestamp = args.sweep_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Nest logs under config name to match run_sweep.py behavior
    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = (Path(args.log_root) / timestamp / args.config).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep ID: {timestamp}")
    
    # 2. Generate Job List
    job_list = []
    global_idx = 0
    # 2. Generate Job List (with Sorting)
    raw_configs = []
    
    # helper to build full dict
    for sweep_set in sets:
        keys = list(sweep_set.keys())
        values = list(sweep_set.values())
        combinations = list(itertools.product(*values))
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            current = base_conf.copy()
            current.update(param_dict)
            raw_configs.append(current)

    # SORTING: travel_steps ascending (if present)
    # Needed for Forage V3 requirements (Travel 50 -> 500)
    if "forage_travel_steps" in raw_configs[0]:
        print("Sorting by forage_travel_steps...")
        raw_configs.sort(key=lambda x: x.get("forage_travel_steps", 0))

    # --- NEW LOGIC: Scan for existing runs ---
    existing_suffixes = set()
    max_existing_id = -1
    
    if base_dir.exists():
        for p in base_dir.glob("run_*"):
            if p.is_dir():
                try:
                    # Parse run_XXX_suffix
                    parts = p.name.split('_', 2)
                    if len(parts) >= 3:
                        run_id = int(parts[1])
                        suffix = parts[2]
                        existing_suffixes.add(suffix)
                        if run_id > max_existing_id:
                            max_existing_id = run_id
                except ValueError:
                    pass

    # Start numbering for *new* runs after the highest existing ID
    # If no existing runs, start at 0
    # If existing runs go up to 179, start at 180
    global_idx = max_existing_id + 1
    print(f"Checking {len(raw_configs)} configs against {len(existing_suffixes)} existing runs.")
    print(f"New runs will start at ID: {global_idx}")

    for current_args in raw_configs:
        # Wrapper for Naming logic (mirroring run_sweep_forage logic)
        parts = []
        
        # 1. Regularization Naming
        if current_args.get("distance_penalty"):
            dist_w = current_args.get("distance_weight", 0)
            parts.append(f"dist-{float(dist_w):.5g}")
        elif current_args.get("l1_weight", 0) > 0:
            l1_w = current_args.get("l1_weight", 0)
            parts.append(f"l1-{float(l1_w):.5g}")
        else:
             parts.append("base")

        # 2. Foraging Naming (Specific)
        if "forage_beta_global" in current_args:
            parts.append(f"beta{current_args['forage_beta_global']}")
        if "forage_alpha_local" in current_args:
            parts.append(f"alpha{current_args['forage_alpha_local']}")
        if "forage_travel_steps" in current_args:
             parts.append(f"trav{current_args['forage_travel_steps']}")
        if "forage_min_block_steps" in current_args:
             parts.append(f"blk{current_args['forage_min_block_steps']}")
        if "forage_temperature" in current_args:
             parts.append(f"temp{current_args['forage_temperature']}")
        
        # Mirroring Run Sweep Forage Logic for Epsilon
        if "forage_eps" in current_args and abs(current_args.get("forage_eps", 0.0)) > 1e-9:
             parts.append(f"eps{current_args['forage_eps']}")
        
        suffix = "_".join(parts)
        
        # CHECK: Does this suffix already exist?
        if suffix in existing_suffixes:
            # Skip adding this job
            # Optional: print(f"Skipping existing: {suffix}")
            continue

        # Create NEW run with next available ID
        run_name = f"run_{global_idx:03d}_{suffix}"
        
        # Command Construction
        cmd_parts = []
        cmd_parts.extend(["--logdir", str(base_dir)])
        cmd_parts.extend(["--run-name", run_name])
        
        for k, v in current_args.items():
            arg_name = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v: cmd_parts.append(arg_name)
            elif isinstance(v, list):
                cmd_parts.append(arg_name)
                cmd_parts.extend([str(item) for item in v])
            else:
                cmd_parts.append(arg_name)
                cmd_parts.append(str(v))
        
        cmd_str = " ".join(cmd_parts)
        job_list.append((run_name, cmd_str))
        global_idx += 1
            
    # 3. Dispatch
    user = getpass.getuser()
    if args.mode == "basic":
        bad, best = scan_nodes(args.cpus)
        run_basic_mode(args, job_list, base_dir, bad, best)
    elif args.mode == "vanilla":
        print(">>> MODE: VANILLA (No Node Selection)")
        print(f"Always excluding known bad nodes: {', '.join(KNOWN_BAD_NODES)}")
        run_basic_mode(args, job_list, base_dir, KNOWN_BAD_NODES, [])  # Exclude known bad nodes
    elif args.mode == "daemon":
        run_daemon_mode(args, job_list, base_dir, user)


if __name__ == "__main__":
    main()
