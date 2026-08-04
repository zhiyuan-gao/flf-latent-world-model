# JEPA-WMs RoboCasa stage-1 reproduction

This stage reproduces the public JEPA-WMs RoboCasa planning evaluation before
adapting anything to RoboCasa365.

## What this benchmark is

- It is **not** the RoboCasa365 target benchmark.
- It uses the authors' `Basile-Terv/robocasa` fork reporting RoboCasa `0.2.0`.
- The evaluated environment is the authors' custom `PnPCounterTop` class,
  added to that fork on 2025-12-15.
- The paper's `Rc-R` and `Rc-Pl` numbers are `reach` and `place` slices of
  this pick-and-place trajectory, with success redefined by distance
  thresholds in the JEPA-WMs wrapper.
- The world model is trained on DROID and evaluated zero-shot in this custom
  RoboCasa environment. No RoboCasa world-model training occurs in stage 1.

## Pinned upstream sources

- `facebookresearch/jepa-wms`: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- `Basile-Terv/robocasa`: `2544dc2e38bb44f5ced80fbc91114a2f7934016a`
- `Basile-Terv/robosuite` master: `63688d5da769b52a77f8dd1da230b8a2602052de`

The JEPA-WMs repository names the two forks but does not pin their commits.
The commits above are the fork revisions predating the released JEPA-WMs code
and containing the authors' custom environment changes.

## Local isolation

- Python environment: `.venv-jepa-wms`
- JEPA-WMs source: `third_party/jepa-wms`
- old RoboCasa fork: `third_party/jepa-wms-robocasa`
- old RoboSuite fork: `third_party/jepa-wms-robosuite`
- data, weights, generated configs and logs: `outputs/jepa_wms_stage1`

This does not replace the existing `third_party/robocasa` and
`third_party/robosuite` checkouts used by the RoboCasa365 work.

## Public artifacts

- Fixed V-JEPA2-AC predictor: `vjepa2_ac_droid.pth.tar`, epoch 315.
- V-JEPA2 ViT-G encoder: native `vitg.pt`, renamed locally to
  `vjepa2_vit_giant.pth` as expected by the upstream configuration.
- RoboCasa dataset: `robocasa/combine_all_im256.hdf5` from
  `facebook/jepa-wms` on Hugging Face.
- Old RoboCasa assets: downloaded by the author fork's asset script.

The downloaded HDF5 is 3,052,915,744 bytes with SHA-256
`2ac3234788e31f79a63ed9f5c964a4e6329e4bdcc45e0b8e15abbd43ad3ed6cd`.
It contains 14 custom `PnPCounterTop` teleoperation trajectories (5,628 total
frames), all described as `pick the liquor from the counter`, rather than
RoboCasa365 task demonstrations. It stores 12-D actions, 256x256 images from
five cameras, simulator states, model XML, episode metadata and subtask segment
labels. The Hugging Face dataset requires the logged-in user to agree to share
contact information before download; access was granted for the local account.

The isolated RoboCasa 0.2.0 environment has been checked at three levels:

- fresh `PnPCounterTop` reset and `robot0_leftview` rendering;
- restoration of an exact dataset model XML and simulator state;
- an end-to-end Reach quick-debug through encoder, predictor, CEM and execution.

The quick-debug ended at 0/1 success, as expected to be non-diagnostic with only
two CEM candidates and two optimizer iterations. The paper setting uses 300
candidates and 15 iterations.

The 32-episode two-GPU run was started successfully and confirmed that both
ranks loaded the exact model and split episodes 0--15 / 16--31. It was then
stopped intentionally before the first action plan completed: one full
300-candidate, 15-iteration CEM optimization had already taken more than five
minutes on each A40. The frozen encoder-plus-predictor stack has about 1.32B
parameters in total; the 305M-parameter predictor is repeatedly evaluated for
the CEM candidates after encoding the current and goal images. No formal
success-rate result was produced, and this interrupted run must not be treated
as an evaluation result.

## Evaluation commands

One-episode, two-sample/two-iteration upstream smoke test:

```bash
.venv-jepa-wms/bin/python scripts/run_jepa_wms_robocasa_stage1.py \
  --subtask reach --quick-debug
```

Full paper-sized Reach evaluation distributed over both local GPUs:

```bash
.venv-jepa-wms/bin/python scripts/run_jepa_wms_robocasa_stage1.py \
  --subtask reach --episodes 32 --devices cuda:0 cuda:1
```

For Place, change `--subtask reach` to `--subtask place`. The local launcher
changes paths and selects the subtask but retains the upstream encoder,
predictor, action mapping, CEM planner, dataset, environment and success
definitions. Multi-GPU execution only partitions episodes across ranks and
reduces the metrics; it does not change the per-episode planner. Optional
decoder plots are disabled by default because the upstream README states they
are not required for planning.
