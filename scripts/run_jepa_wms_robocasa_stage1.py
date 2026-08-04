#!/usr/bin/env python3
"""Run the upstream JEPA-WMs V-JEPA2-AC RoboCasa evaluation locally.

The upstream paper evaluates the DROID-trained fixed V-JEPA2-AC predictor in
its custom RoboCasa PnPCounterTop environment.  This launcher leaves the
upstream model, dataset, planner, and environment code untouched.  It only
resolves local paths, selects the reach/place subtask, and optionally disables
the decoder-only visualizations that do not affect planning success.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM = WORKSPACE / "third_party" / "jepa-wms"
STAGE_ROOT = WORKSPACE / "outputs" / "jepa_wms_stage1"
BASE_CONFIG = (
    UPSTREAM
    / "configs"
    / "evals"
    / "simu_env_planning"
    / "rcasa_custom"
    / "vj2ac"
    / "reach_L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0_ep32_decode.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=("reach", "place"), default="reach")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--quick-debug",
        action="store_true",
        help="Use the upstream two-sample/two-iteration one-episode smoke mode.",
    )
    parser.add_argument(
        "--with-decoder-plots",
        action="store_true",
        help="Load the optional image decoder and emit upstream diagnostic plots.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Distribute evaluation episodes across these devices, e.g. cuda:0 cuda:1.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[dict, Path]:
    with BASE_CONFIG.open() as stream:
        config = yaml.safe_load(stream)

    log_root = STAGE_ROOT / "logs" / f"vj2ac_fixed_{args.subtask}"
    checkpoint_root = STAGE_ROOT / "checkpoints"
    encoder_root = STAGE_ROOT / "oss_checkpoints"

    config["folder"] = str(log_root)
    config["checkpoint_folder"] = str(checkpoint_root)
    config["model_kwargs"]["checkpoint"] = "vjepa2_ac_droid.pth.tar"
    config["model_kwargs"]["pretrain_kwargs"]["visual_encoder"][
        "pretrain_enc_path"
    ] = str(encoder_root / "vjepa2_opensource" / "vjepa2_vit_giant.pth")
    config["task_specification"]["env"]["subtask"] = args.subtask
    config["meta"]["eval_episodes"] = args.episodes
    config["meta"]["quick_debug"] = args.quick_debug
    config["tag"] = f"stage1_{args.subtask}_{'smoke' if args.quick_debug else f'ep{args.episodes}'}"

    if not args.with_decoder_plots:
        config["logging"]["optional_plots"] = False
        config["planner"]["decode_each_iteration"] = False
        config["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {
            "architectures": {},
            "pretrain_dec_path": None,
        }

    config_dir = STAGE_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_config = config_dir / f"vj2ac_fixed_{args.subtask}_{'smoke' if args.quick_debug else f'ep{args.episodes}'}.yaml"
    with output_config.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config, output_config


def run_rank(rank: int, world_size: int, devices: list[str], config_path: Path) -> None:
    # The upstream wrapper otherwise derives this path from JEPAWM_HOME, where
    # this workspace also contains a newer RoboCasa365 checkout.  Point only
    # the old evaluation process at the isolated author fork.
    from evals.simu_env_planning.envs import robocasa as robocasa_env

    robocasa_env.BASE_ASSET_ROOT_PATH = str(
        WORKSPACE
        / "third_party"
        / "jepa-wms-robocasa"
        / "robocasa"
        / "models"
        / "assets"
        / "objects"
    )

    from evals.main import process_main

    upstream_args = SimpleNamespace(
        checkpoint=None,
        model_name=None,
        batch_size=None,
        folder=None,
        use_fsdp=False,
    )
    process_main(
        args=upstream_args,
        rank=rank,
        fname=str(config_path),
        world_size=world_size,
        devices=devices,
    )


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    os.environ.setdefault("JEPAWM_DSET", str(STAGE_ROOT / "data"))
    os.environ.setdefault("JEPAWM_LOGS", str(STAGE_ROOT / "logs"))
    os.environ.setdefault("JEPAWM_CKPT", str(STAGE_ROOT / "checkpoints"))
    os.environ.setdefault("JEPAWM_OSSCKPT", str(STAGE_ROOT / "oss_checkpoints"))
    os.environ.setdefault("JEPAWM_HOME", str(WORKSPACE / "third_party"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    _, config_path = build_config(args)
    devices = args.devices if args.devices is not None else [args.device]
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")

    if len(devices) == 1:
        run_rank(rank=0, world_size=1, devices=devices, config_path=config_path)
        return

    mp.set_start_method("spawn", force=True)
    processes = [
        mp.Process(
            target=run_rank,
            args=(rank, len(devices), devices, config_path),
            name=f"jepa-wms-eval-rank-{rank}",
        )
        for rank in range(len(devices))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [(process.name, process.exitcode) for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"Distributed evaluation failed: {failures}")


if __name__ == "__main__":
    main()
