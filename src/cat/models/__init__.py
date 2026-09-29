"""Optional Hugging Face / PEFT backend; imports do not load model weights."""
from .hf import HFModelConfig, LoRASettings, load_lora_model, save_lora_adapter

__all__ = ["HFModelConfig", "LoRASettings", "load_lora_model", "save_lora_adapter"]
