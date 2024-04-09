import os
import logging

eval_logger = logging.getLogger("lmms-eval")

AVAILABLE_MODELS = {
    "llava": "Llava",
    "qwen_vl": "Qwen_VL",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        eval_logger.error(f"Error importing {model_class}: {e}")


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
