#!/usr/bin/env python3
"""Print experiment configurations; never launch a job."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    for row in json.loads((root/'configs/paper_experiments.json').read_text())['experiments']:
        print(f"\n{row['id']} — {row['paper']}")
        for config in row['configs']:
            print(f"  python scripts/{row['trainer']} --config {config} --dry-run")
