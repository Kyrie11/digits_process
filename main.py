from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import inspect_command, run_command
from src.reporting import report_command
from src.utils import load_yaml


def load_config_with_overrides(config_path: str, mat_path: str | None, output_dir: str | None, device: str | None):
    cfg = load_yaml(config_path)
    cfg.setdefault("data", {})
    cfg.setdefault("experiment", {})
    if mat_path is not None:
        cfg["data"]["mat_path"] = mat_path
    if output_dir is not None:
        cfg["experiment"]["output_dir"] = output_dir
    if device is not None:
        cfg["experiment"]["device"] = device
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digit preprocessing course project runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser):
        subparser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
        subparser.add_argument("--mat_path", type=str, default=None, help="Override path to digits4000.mat.")
        subparser.add_argument("--output_dir", type=str, default=None, help="Override experiment output directory.")
        subparser.add_argument("--device", type=str, default=None, help="cpu / cuda / auto")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect dataset and save preview assets.")
    add_common_args(inspect_parser)

    run_parser = subparsers.add_parser("run", help="Run full experiment grid.")
    add_common_args(run_parser)

    report_parser = subparsers.add_parser("report", help="Generate report tables and figures from a finished run.")
    add_common_args(report_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.mat_path, args.output_dir, args.device)

    command = args.command
    if command == "inspect":
        inspect_command(cfg)
    elif command == "run":
        run_command(cfg)
    elif command == "report":
        report_command(cfg)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
