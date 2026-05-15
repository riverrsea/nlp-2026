from __future__ import annotations

import argparse
from typing import Any, Dict

from config import load_experiment_config
from trainers import get_trainer_class
from utils import ensure_directories, resolve_device, setup_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry point for Chinese text classification experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file, for example cfg/textcnn_random.yaml.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Run mode: train or test.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory of the raw dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory for outputs such as results and checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device. Use `auto` to select cuda or cpu automatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path used in mode=test.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Pretrained word vector path for TextCNN-Pretrained.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Pretrained model name or local path for BERT-based models.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=None,
        help="Logging level such as INFO or DEBUG.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "mode": args.mode,
        "device": args.device,
        "paths": {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
        },
    }

    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.checkpoint is not None:
        overrides["checkpoint"] = args.checkpoint
    if args.pretrained_path is not None:
        overrides["pretrained_path"] = args.pretrained_path
    if args.model_name_or_path is not None:
        overrides["model_name_or_path"] = args.model_name_or_path
    if args.log_level is not None:
        overrides["logging"] = {"level": args.log_level}

    return overrides


def prepare_runtime(config: Dict[str, Any], logger) -> None:
    ensure_directories(
        [
            config["paths"]["output_dir"],
            config["paths"]["data_output_dir"],
            config["paths"]["figures_dir"],
            config["paths"]["results_dir"],
            config["paths"]["predictions_dir"],
            config["paths"]["checkpoints_dir"],
            config["paths"]["experiment_checkpoint_dir"],
        ]
    )
    set_seed(int(config["seed"]))
    config["device"] = resolve_device(str(config.get("device", "auto")))

    logger.info("Experiment name: %s", config["experiment_name"])
    logger.info("Model name: %s", config["model_name"])
    logger.info("Mode: %s", config["mode"])
    logger.info("Resolved device: %s", config["device"])
    logger.info("Output directory: %s", config["paths"]["output_dir"])


def run() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(args.config, overrides=build_overrides(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    logger = setup_logging(config["logging"]["level"], name="main")

    try:
        prepare_runtime(config, logger)
        trainer_class = get_trainer_class(str(config["model_name"]))
        trainer = trainer_class(config=config, logger=logger)

        if config["mode"] == "train":
            result = trainer.train()
        else:
            result = trainer.test()

        logger.info("Finished with status: %s", result.get("status", "unknown"))
        return 0
    except (FileNotFoundError, ImportError, ValueError) as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
