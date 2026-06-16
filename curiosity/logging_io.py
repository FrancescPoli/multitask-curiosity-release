from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import csv
import matplotlib.pyplot as plt

def ensure_run_dir(cfg) -> Path:
    run_dir = Path(cfg.logdir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def savefig_show(path: Path, fig=None):
    (fig or plt.gcf()).tight_layout()
    (fig or plt.gcf()).savefig(path)
    print(f"[plot saved] {path}")

def write_csv_rows(path: Path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            pass
        return
    key_set = set()
    for r in rows:
        key_set.update(r.keys())
    preferred = ['step','regime','task','arm','p_arm','prob','reward','loss_pre','loss_post','acc']
    fieldnames = [k for k in preferred if k in key_set] + [k for k in sorted(key_set) if k not in preferred]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})

@dataclass
class CsvLogger:
    path: Path
    fieldnames: Optional[List[str]] = None
    _rows: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, row: Dict[str, Any]) -> None:
        self._rows.append(dict(row))

    def flush(self) -> None:
        write_csv_rows(self.path, self._rows)

@dataclass
class PolicyLogger:
    path: Path
    _rows: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, row: Dict[str, Any]) -> None:
        self._rows.append(dict(row))

    def flush(self) -> None:
        write_csv_rows(self.path, self._rows)
