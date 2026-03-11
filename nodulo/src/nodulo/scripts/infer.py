from __future__ import annotations

import argparse
from pathlib import Path

from nodulo.config import AppConfig
from nodulo.training.pipeline import infer_from_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-phase lung nodule classification and localization inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing test images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where CSV outputs will be written.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Path to the classifier soup checkpoint.")
    parser.add_argument("--localizer-checkpoint", type=Path, required=True, help="Path to the localization soup checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_yaml(args.config).raw
    outputs = infer_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        classifier_checkpoint=args.classifier_checkpoint,
        localizer_checkpoint=args.localizer_checkpoint,
        config=config,
    )
    print(outputs)


if __name__ == "__main__":
    main()
