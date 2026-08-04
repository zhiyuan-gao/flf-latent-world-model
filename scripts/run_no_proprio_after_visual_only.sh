#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/.." && pwd)"
current_session=vjepa2_native256_visual_only_queue
current_run="$project/outputs/single_step_dynamics/formal_native256_visual_only_100_per_task_stride1_b128_seed_0"
next_run="$project/outputs/single_step_dynamics/formal_native256_no_proprio_100_per_task_stride1_b128_seed_0"

cd "$project"
mkdir -p "$next_run"

printf 'Queued no-proprio run at %s\n' "$(date --iso-8601=seconds)"
printf 'Waiting for tmux session %s to finish.\n' "$current_session"
while tmux has-session -t "$current_session" 2>/dev/null; do
  sleep 30
done

printf 'Visual-only conditioned session ended at %s; validating completion summary.\n' \
  "$(date --iso-8601=seconds)"
upstream_status=failed
if .venv-robocasa-gr00t/bin/python - "$current_run/summary.json" <<'PY'
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
    "architecture": "single_step_visual_only_proprio_conditioned_t4_v1",
    "predicts_future_proprio": False,
    "uses_current_proprio_condition": True,
    "proprio_weight": 0.0,
}
observed = {key: summary.get(key) for key in expected}
if observed != expected:
    raise SystemExit(
        f"Refusing to launch: conditioned visual-only run did not complete the "
        f"fixed protocol; expected={expected}, observed={observed}"
    )
print("Conditioned visual-only run completed the fixed protocol.", flush=True)
PY
then
  upstream_status=complete
else
  printf 'Conditioned visual-only run failed or is incomplete; continuing with fallback policy.\n' >&2
fi

wait_for_idle_gpus() {
  local check=1
  local max_checks=60
  local gpu_pids
  local trainer_pids
  while (( check <= max_checks )); do
    gpu_pids="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
        sed '/^[[:space:]]*$/d'
    )"
    trainer_pids="$(
      .venv-robocasa-gr00t/bin/python - <<'PY'
import os
from pathlib import Path

target = "scripts/train_single_step_dynamics.py"
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        executable = Path(os.readlink(proc / "exe")).name.lower()
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        argv = [arg.decode(errors="replace") for arg in argv if arg]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if not executable.startswith("python"):
        continue
    if any(arg == target or arg.endswith("/" + target) for arg in argv):
        print(proc.name)
PY
    )"
    if test -z "$gpu_pids" && test -z "$trainer_pids"; then
      return 0
    fi
    if (( check == 1 || check % 10 == 0 )); then
      printf 'Waiting for idle GPUs and no residual trainer (%d/%d); GPU PIDs: %s; trainer PIDs: %s\n' \
        "$check" "$max_checks" "${gpu_pids:-none}" "${trainer_pids:-none}"
    fi
    sleep 30
    ((check += 1))
  done
  return 1
}

if ! wait_for_idle_gpus; then
  printf 'GPUs or residual trainers did not become idle within 30 minutes; refusing overlap.\n' >&2
  exit 1
fi

.venv-robocasa-gr00t/bin/python - \
  "$next_run/queue_context.json" "$upstream_status" "$current_run/summary.json" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output = Path(sys.argv[1])
status = sys.argv[2]
upstream_summary = Path(sys.argv[3])
record = {
    "recorded_at": datetime.now().astimezone().isoformat(),
    "upstream_status": status,
    "upstream_summary": str(upstream_summary),
    "fallback_launch": status != "complete",
}
output.write_text(json.dumps(record, indent=2) + "\n")
print(f"Recorded queue context: {record}", flush=True)
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
    --no-proprio-target --no-proprio-input \
    --max-epochs 25 --min-epochs 25 \
    --disable-early-stopping --checkpoint-epochs 5 10 15 20 25 \
    --output-dir "$next_run"
}

if has_retained_progress; then
  printf 'Refusing to overwrite retained no-proprio progress in %s\n' "$next_run" >&2
  exit 1
fi

max_attempts=3
attempt=1
last_status=1
while (( attempt <= max_attempts )); do
  printf 'Launching no-proprio attempt %d/%d with upstream_status=%s at %s\n' \
    "$attempt" "$max_attempts" "$upstream_status" "$(date --iso-8601=seconds)"
  if run_training; then
    last_status=0
  else
    last_status=$?
  fi
  if (( last_status == 0 )); then
    printf 'No-proprio training completed successfully at %s\n' \
      "$(date --iso-8601=seconds)"
    exit 0
  fi
  printf 'No-proprio attempt %d exited with status %d.\n' \
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

printf 'No-proprio startup failed after %d attempts.\n' "$max_attempts" >&2
exit "$last_status"
