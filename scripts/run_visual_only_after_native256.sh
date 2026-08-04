#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/.." && pwd)"
current_session=vjepa2_native256_train
current_run="$project/outputs/single_step_dynamics/formal_native256_100_per_task_stride1_b128_seed_0"
next_run="$project/outputs/single_step_dynamics/formal_native256_visual_only_100_per_task_stride1_b128_seed_0"

cd "$project"
mkdir -p "$next_run"

printf 'Queued visual-only run at %s\n' "$(date --iso-8601=seconds)"
printf 'Waiting for tmux session %s to finish.\n' "$current_session"
while tmux has-session -t "$current_session" 2>/dev/null; do
  sleep 30
done

printf 'Current session ended at %s; validating completion summary.\n' "$(date --iso-8601=seconds)"
.venv-robocasa-gr00t/bin/python - "$current_run/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
if not summary_path.is_file():
    raise SystemExit(f"Refusing to launch: missing {summary_path}")
summary = json.loads(summary_path.read_text())
expected = {
    "completed_epochs": 25,
    "completed_steps": 44_425,
    "stopped_early": False,
    "encoder": "vjepa2_native_256",
    "architecture": "single_step_visual_proprio_t4_v1",
    "proprio_weight": 0.005,
}
observed = {key: summary.get(key) for key in expected}
if observed != expected:
    raise SystemExit(
        f"Refusing to launch: current run did not complete the fixed protocol; "
        f"expected={expected}, observed={observed}"
    )
print("Current joint-target run completed the fixed protocol.", flush=True)
PY

has_retained_progress() {
  test -e "$next_run/history.json" ||
    test -e "$next_run/summary.json" ||
    find "$next_run" -maxdepth 1 -type f -name 'epoch_*.pt' -print -quit |
      grep -q .
}

run_training() {
  env CUDA_VISIBLE_DEVICES=0,1 .venv-robocasa-gr00t/bin/python \
    -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/train_single_step_dynamics.py \
    --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
    --feature-root outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256 \
    --seed 0 --batch-size 128 \
    --learning-rate 3e-4 --weight-decay 0.05 --warmup-ratio 0.05 \
    --no-proprio-target --max-epochs 25 --min-epochs 25 \
    --disable-early-stopping --checkpoint-epochs 5 10 15 20 25 \
    --output-dir "$next_run"
}

if has_retained_progress; then
  printf 'Refusing to overwrite retained visual-only progress in %s\n' "$next_run" >&2
  exit 1
fi

max_attempts=3
attempt=1
last_status=1
while (( attempt <= max_attempts )); do
  printf 'Launching visual-only attempt %d/%d at %s\n' \
    "$attempt" "$max_attempts" "$(date --iso-8601=seconds)"
  if run_training; then
    last_status=0
  else
    last_status=$?
  fi
  if (( last_status == 0 )); then
    printf 'Visual-only training completed successfully at %s\n' \
      "$(date --iso-8601=seconds)"
    exit 0
  fi
  printf 'Visual-only attempt %d exited with status %d.\n' \
    "$attempt" "$last_status" >&2
  if has_retained_progress; then
    printf 'Retained epoch/history artifacts exist; refusing an unsafe restart.\n' >&2
    exit "$last_status"
  fi
  if (( attempt == max_attempts )); then
    break
  fi
  printf 'No retained epoch/checkpoint progress; retrying in 120 seconds.\n'
  sleep 120
  ((attempt += 1))
done

printf 'Visual-only startup failed after %d attempts.\n' "$max_attempts" >&2
exit "$last_status"
