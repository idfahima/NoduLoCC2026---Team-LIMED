from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "tf32"
if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
    torch.backends.cudnn.conv.fp32_precision = "tf32"

from nodulo.config import AppConfig
from nodulo.training.pipeline import train_two_phase_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-phase RAD-DINO lung nodule pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_yaml(args.config).raw
    checkpoints = train_two_phase_pipeline(config)
    print(checkpoints)


if __name__ == "__main__":
    main()
