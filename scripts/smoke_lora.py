#!/usr/bin/env python3
"""Test the real Transformers/PEFT loader using a tiny locally created Mistral."""
from cat.testing.lora_smoke import run_lora_smoke

if __name__ == "__main__":
    run_lora_smoke()
