
"""
Path Handling Utility
=====================

Centralizes path resolution for running analysis scripts both locally (Windows/Z-drive) 
and on the cluster (Linux/Imaging FS).

Usage:
    from analysis.utils.paths import get_logs_dir, get_data_dir

    logs_dir = get_logs_dir()
    data_dir = get_data_dir()
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """
    Returns the project root directory.
    Assumes this file is in <project_root>/analysis/utils/paths.py
    """
    # This file is analysis/utils/paths.py -> parent -> parent -> parent = root
    return Path(__file__).resolve().parent.parent.parent


def get_logs_dir():
    """
    Canonical logs directory — single source of truth for the whole repo
    (training, sweeps, and analysis all resolve through here; the repo-root
    paths.py is a thin shim that re-exports this function).

    Priority:
    1. LOGS_DIR environment variable    -> set this to send LOCAL runs anywhere
                                           (both training and analysis honor it).
    2. Windows network mount (cluster store): Z:/fp02/logs   (matches the sweep tree).
    3. Windows local fallback when Z: is not mounted: C:/Users/fp02/OneDrive/Documenti/logs.
    4. Cluster (Linux): /imaging/astle/fp02/logs, then the repo's sibling ../logs.
    5. Last resort (ad-hoc local run, nothing else found): <project_root>/logs.
    """
    # 1. Explicit override — the clean way to run models locally.
    env_logs = os.environ.get("LOGS_DIR")
    if env_logs:
        return Path(env_logs).resolve()

    # 2-3. Windows
    if sys.platform == "win32":
        z_logs = Path("Z:/fp02/logs")            # canonical (Z:/fp02/logs/sweep/...)
        if z_logs.exists():
            return z_logs
        onedrive_logs = Path("C:/Users/fp02/OneDrive/Documenti/logs")
        if onedrive_logs.exists():
            return onedrive_logs

    # 4. Cluster (Linux): canonical path, then repo-sibling ../logs.
    project_root = get_project_root()
    cluster_logs = Path("/imaging/astle/fp02/logs")
    if cluster_logs.exists():
        return cluster_logs
    sibling_logs = project_root.parent / "logs"
    if sibling_logs.exists():
        return sibling_logs

    # 5. Ad-hoc local fallback (created by the caller on first write).
    return project_root / "logs"

def get_data_dir():
    """
    Returns the path to the 'analysis/Network_analysis/Data' directory or equivalent external data source.
    """
    # 1. Local Z-drive for large data (lifespan_project_data_v7_3.mat)
    z_data = Path("Z:/fp02/wGNM")
    if z_data.exists():
        return z_data
        
    # 2. Cluster large data path
    cluster_data = Path("/imaging/astle/fp02/wGNM") 
    if cluster_data.exists():
        return cluster_data

    # 3. Default internal data (analysis/Network_analysis/Data)
    return get_project_root() / "analysis" / "Network_analysis" / "Data"

def resolve_sweep_dir(sweep_name_or_path):
    """
    Resolves a sweep path given a name (e.g., 'forage_v5') or a full path.
    """
    
    path = Path(sweep_name_or_path)
    if path.exists():
        return path

    # If absolute exist, return
    if path.exists():
        return path
    
    # Try finding it relative to project root
    project_root = get_project_root()
    relative_path = project_root / path
    if relative_path.exists():
        return relative_path
        
    # Handle case where user provided "logs/..." but logs is a sibling
    parts = path.parts
    if parts and parts[0] == "logs":
        # Check relative to logs dir
        # "logs/sweep/X" -> get_logs_dir() / "sweep/X"
        remainder = Path(*parts[1:])
        logs_dir = get_logs_dir()
        rehomed = logs_dir / remainder
        if rehomed.exists():
            return rehomed
            
    # Also check directly inside logs dir if it's just a name
    logs_dir = get_logs_dir()
    direct = logs_dir / path
    if direct.exists():
        return direct
        
    sweep_in_logs = logs_dir / "sweep" / path
    if sweep_in_logs.exists():
        return sweep_in_logs

    return path # Return path even if not found, let caller error out
