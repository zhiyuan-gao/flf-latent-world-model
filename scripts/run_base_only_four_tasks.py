#!/usr/bin/env python3
"""Run the frozen GR00T N1.5 RoboCasa365 baseline on the MVP task subset.

This is a project-owned, task-filtered entry point around the official
robocasa-benchmark/Isaac-GR00T policy server and simulation client.  It keeps
the benchmark fork unmodified while making it impossible to accidentally run
the other twelve Composite-Seen tasks during MVP smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon


MVP_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_model = (
        project_root
        / "checkpoints"
        / "gr00t_n1-5_composite_seen_target_posttraining"
        / "checkpoint-60000"
    )
    default_output = project_root / "outputs" / "base_only_official_chunk16"

    parser = argparse.ArgumentParser(
        description="Evaluate the frozen official GR00T N1.5 baseline on the four MVP tasks."
    )
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--tasks", nargs="+", default=list(MVP_TASKS))
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Execute this many actions before replanning; 16 is the official evaluation default.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional smoke-test override. By default each task uses its official horizon.",
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run a task even when its stats.json already exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.model_path = args.model_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.model_path}")
    if args.n_episodes < 1 or args.n_envs < 1 or args.n_action_steps < 1:
        raise ValueError("n-episodes, n-envs, and n-action-steps must all be positive")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        raise ValueError("max-episode-steps must be positive when supplied")

    requested = list(dict.fromkeys(args.tasks))
    unknown = sorted(set(requested) - set(MVP_TASKS))
    if unknown:
        raise ValueError(
            f"Tasks outside the controlled MVP whitelist are not allowed: {unknown}. "
            f"Allowed tasks: {list(MVP_TASKS)}"
        )
    missing = sorted(set(requested) - set(TASK_SET_REGISTRY["composite_seen"]))
    if missing:
        raise ValueError(f"Tasks missing from RoboCasa composite_seen registry: {missing}")
    args.tasks = requested


def build_policy(args: argparse.Namespace) -> Gr00tPolicy:
    data_config = DATA_CONFIG_MAP["panda_omron"]
    return Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=4,
        device=args.device,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline": "GR00T N1.5 RoboCasa365 Composite-Seen target post-training",
        "checkpoint": str(args.model_path),
        "checkpoint_step": 60000,
        "data_config": "panda_omron",
        "embodiment_tag": "new_embodiment",
        "denoising_steps": 4,
        "split": args.split,
        "tasks": args.tasks,
        "n_episodes": args.n_episodes,
        "n_envs": args.n_envs,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps_override": args.max_episode_steps,
        "video_enabled": not args.no_video,
        "pid": os.getpid(),
        "started_at_unix": time.time(),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)

    print("Loading frozen GR00T policy...", flush=True)
    policy = build_policy(args)
    server = RobotInferenceServer(policy, port=args.port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    client = SimulationInferenceClient(host="localhost", port=args.port)
    if not client.ping():
        raise RuntimeError(f"Inference server did not answer on port {args.port}")

    failed_tasks: list[str] = []
    for task in args.tasks:
        task_dir = args.output_dir / "evals" / args.split / task
        stats_path = task_dir / "stats.json"
        if stats_path.exists() and not args.overwrite:
            print(f"{task}: {stats_path} already exists; skipping.", flush=True)
            continue

        official_horizon = int(get_task_horizon(task))
        horizon = args.max_episode_steps or official_horizon
        video_dir = None if args.no_video else str(task_dir)
        config = SimulationConfig(
            env_name=f"robocasa/{task}",
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps,
                max_episode_steps=horizon,
            ),
        )

        started = time.time()
        print(
            f"Running {task}: episodes={args.n_episodes}, action_chunk={args.n_action_steps}, "
            f"horizon={horizon} (official={official_horizon})",
            flush=True,
        )
        try:
            _, successes = client.run_simulation(config)
        except Exception as exc:  # keep the remaining tasks runnable and preserve the error
            failed_tasks.append(task)
            write_json(
                task_dir / "error.json",
                {
                    "task": task,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.time() - started,
                },
            )
            print(f"{task}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue

        successes = [bool(value) for value in successes[: args.n_episodes]]
        stats = {
            "task": task,
            "num_episodes": len(successes),
            "episode_successes": successes,
            "success_rate": float(np.mean(successes)),
            "split": args.split,
            "n_action_steps": args.n_action_steps,
            "max_episode_steps": horizon,
            "official_horizon": official_horizon,
            "elapsed_seconds": time.time() - started,
        }
        write_json(stats_path, stats)
        print(f"{task}: success_rate={stats['success_rate']:.3f}; saved {stats_path}", flush=True)

    manifest["completed_at_unix"] = time.time()
    manifest["failed_tasks"] = failed_tasks
    write_json(args.output_dir / "run_manifest.json", manifest)
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}", file=sys.stderr)
        return 1
    print(f"Finished controlled four-task baseline run: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
