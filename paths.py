"""
Repo-root shim for `from paths import get_logs_dir`.

The single source of truth lives in analysis/utils/paths.py. This shim only
exists so the training/sweep entry points (run.py, experiments/run_sweep*.py),
which add the repo root to sys.path and do `from paths import get_logs_dir`,
keep resolving the *same* canonical logs directory as the analysis side.
"""
from analysis.utils.paths import get_logs_dir

__all__ = ["get_logs_dir"]

if __name__ == "__main__":
    print(f"Resolved Log Directory: {get_logs_dir()}")
