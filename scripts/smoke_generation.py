#!/usr/bin/env python3
"""Local generation integration using a tiny randomly initialized model."""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from cat.testing.generation_smoke import run_generation_smoke

if __name__ == "__main__":
    run_generation_smoke()
