"""Sweep driver that launches multiple training runs from partial configs."""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge one JSON-like mapping into another
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file from disk
    """
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    """
    Write a JSON file, create parent directories if needed
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def filesystem_safe_string(name: str) -> str:
    """
    Convert a filename or experiment name into a filesystem-safe slug
    """
    name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "experiment"


def list_config_files(config_dir: Path) -> List[Path]:
    """
    Return all JSON config files in deterministic order
    """
    return sorted([p for p in config_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])


def make_experiment_output_dir(base_output_dir: Path, config_file: Path, partial_cfg: Dict[str, Any],
                               index: int) -> Path:
    """
    Build the per-experiment output directory name
    """
    experiment_name = partial_cfg.get("EXPERIMENT_NAME") or filesystem_safe_string(config_file.name)
    prefix = f"{index:03d}_{filesystem_safe_string(experiment_name)}"
    return base_output_dir / prefix


def run_one_experiment(
        trainer_script: Path,
        resolved_config_path: Path,
        log_path: Path,
        python_executable: str,
        extra_env: Dict[str, str],
) -> int:
    """
    Run one experiment as a subprocess and capture stdout/stderr to a log
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [python_executable, str(trainer_script), "--config", str(resolved_config_path)]

    with open(log_path, "w") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **extra_env},
        )

    return int(process.returncode)


def main() -> None:
    """
    CLI entrypoint
    """
    # --- argument parsing ---
    parser = argparse.ArgumentParser(
        description="Sequentially run light-curve training experiments from a default config and a directory of partial configs."
    )
    parser.add_argument("--default-config", type=str, default="default.json", help="Path to the full/default JSON config.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Path to the directory containing configs.")
    parser.add_argument("--trainer-script", type=str, default="experiment.py", help="Path to the trainer script.")
    parser.add_argument("--sweep-output-dir", type=str, default="results",
                        help="Directory where resolved configs, logs, and summary are written.")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable used for each training subprocess.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running later configs if one experiment fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write resolved configs and summary without launching training.")
    args = parser.parse_args()

    # configuration setup
    default_config_path = Path(args.default_config).resolve()
    config_dir = Path(args.config_dir).resolve()
    trainer_script = Path(args.trainer_script).resolve()
    sweep_output_dir = Path(args.sweep_output_dir).resolve()

    # error handling
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_config_path}")
    if not config_dir.exists() or not config_dir.is_dir():
        raise NotADirectoryError(f"Config directory not found: {config_dir}")
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    default_cfg = load_json(default_config_path)
    config_files = list_config_files(config_dir)
    if not config_files:
        raise FileNotFoundError(f"No .json config files found in {config_dir}")

    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_output_dir / "sweep_summary.json"
    summary_rows: List[Dict[str, Any]] = []

    sweep_start = time.perf_counter()
    sweep_start_local = time.strftime("%Y-%m-%d %H:%M:%S")

    # --- iterate over experiment configuration ---
    for index, config_file in enumerate(config_files, start=1):
        # setup
        partial_cfg = load_json(config_file)
        resolved_cfg = deep_update(default_cfg, partial_cfg)

        base_output_dir = Path(
            resolved_cfg.get("OUTPUT_DIR", default_cfg.get("OUTPUT_DIR", "results/config_sweep_runs"))).resolve()
        experiment_output_dir = make_experiment_output_dir(base_output_dir, config_file, partial_cfg, index)
        resolved_cfg["OUTPUT_DIR"] = str(experiment_output_dir)
        resolved_cfg["EXPERIMENT_NAME"] = resolved_cfg.get("EXPERIMENT_NAME") or filesystem_safe_string(config_file.name)

        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = experiment_output_dir / "resolved_config.json"
        log_path = experiment_output_dir / "run.log"
        save_json(resolved_config_path, resolved_cfg)
        save_json(experiment_output_dir / "partial_config_used.json", partial_cfg)

        row = {
            "index": index,
            "config_file": str(config_file),
            "experiment_name": resolved_cfg["EXPERIMENT_NAME"],
            "output_dir": str(experiment_output_dir),
            "resolved_config": str(resolved_config_path),
            "log_path": str(log_path),
            "status": "pending",
            "start_time_local": None,
            "end_time_local": None,
            "duration_seconds": None,
            "returncode": None,
        }

        if args.dry_run:
            row["status"] = "dry_run"
            summary_rows.append(row)
            save_json(summary_path, {"sweep_start_local": sweep_start_local, "experiments": summary_rows})
            print(f"[dry-run] Prepared {config_file.name} -> {experiment_output_dir}")
            continue

        print(f"[{index}/{len(config_files)}] Running {config_file.name} -> {experiment_output_dir}")
        row["status"] = "running"
        row["start_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_start = time.perf_counter()
        save_json(summary_path, {"sweep_start_local": sweep_start_local, "experiments": summary_rows + [row]})

        # --- run the experiment ---
        returncode = run_one_experiment(
            trainer_script=trainer_script,
            resolved_config_path=resolved_config_path,
            log_path=log_path,
            python_executable=args.python,
            extra_env={},
        )

        # --- run metrics ---
        duration_seconds = float(time.perf_counter() - experiment_start)
        row["end_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        row["duration_seconds"] = duration_seconds
        row["duration_minutes"] = duration_seconds / 60.0
        row["duration_hours"] = duration_seconds / 3600.0
        row["returncode"] = returncode
        row["status"] = "completed" if returncode == 0 else "failed"

        save_json(
            experiment_output_dir / "driver_timing.json",
            {
                "status": row["status"],
                "start_time_local": row["start_time_local"],
                "end_time_local": row["end_time_local"],
                "duration_seconds": row["duration_seconds"],
                "duration_minutes": row["duration_minutes"],
                "duration_hours": row["duration_hours"],
                "returncode": returncode,
                "log_path": str(log_path),
            },
        )

        summary_rows.append(row)
        save_json(
            summary_path,
            {
                "sweep_start_local": sweep_start_local,
                "sweep_elapsed_seconds_so_far": float(time.perf_counter() - sweep_start),
                "experiments": summary_rows,
            },
        )

        if returncode != 0:
            print(f"Experiment failed with return code {returncode}. Log: {log_path}")
            if not args.continue_on_error:
                break

    sweep_duration = float(time.perf_counter() - sweep_start)
    completed = sum(1 for row in summary_rows if row["status"] == "completed")
    failed = sum(1 for row in summary_rows if row["status"] == "failed")
    save_json(
        summary_path,
        {
            "sweep_start_local": sweep_start_local,
            "sweep_end_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sweep_duration_seconds": sweep_duration,
            "sweep_duration_minutes": sweep_duration / 60.0,
            "sweep_duration_hours": sweep_duration / 3600.0,
            "completed": completed,
            "failed": failed,
            "experiments": summary_rows,
        },
    )
    print(f"Sweep summary written to: {summary_path}")


if __name__ == "__main__":
    main()
