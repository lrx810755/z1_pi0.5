# Independent ACT handoff

The pipeline is staged here. GPU smoke and training have **not** been run. All outputs are local to this directory; the original500 episodes, crops, checkpoints, and production source remain intact.

The current collection target is19 distinct failed rollout recoveries (6 current8Hz,13 prior10Hz;18 distinct seeds). The target was revised from20 after source screening. Completion may accept16–20 only when the independent verifier declares the exact final count; no padding, copying or success-only admission is allowed.

## Required accepted dataset

Default source: `/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028` (override with `Z1_PUSH_RECOVERY_DATA`).

`completion_verified.json` must contain `passed:true`, `source_type:"actual_policy_push_recovery"`, `accepted_count`, and `episodes` as absolute paths. The list must exactly match all `episode_[0-9]*` directories. Each correction must independently declare `accepted:true`, `recovery_complete:true`, `source_type:"actual_policy_push_recovery"`, and unique `source_identity` (model/source rollout/takeover time). Merely setting task success cannot pass.

Each accepted episode preserves the prior raw format:

- `frames.jsonl`: aligned `frame_index`, absolute `simulation_time`, relative timestamp, action/state7, cup task/target, and three relative JPEG image paths.
- `episode_summary.json`, `episode_start.json`, `correction.json`, `gripper_labels.json`, `training_mask.json`, `physics_status.jsonl`.
- `gripper_labels.json`: aligned binary `labels`.
- `training_mask.json`: aligned boolean `expert_frame_mask`. Policy failure and human intervention frames are false. Eligibility is recomputed for horizon30; old horizon16 declarations do not control this round.
- `correction.json`: `source_grid` is the **original initial cup layout**, not the pushed cup position; `recovery_intervals_s:[{"start":absolute_simulation_time,"end":absolute_simulation_time}]`; `dynamic_samples`, `table_contact:false`, `dynamic_violations:[]`, final physical success evidence matching prior schema. Summary `scene_randomization.target_grid` and seed must preserve source grouping. Root collection sets these fields from the original source layout. Grid uses the existing500 data convention (first coordinate0..4, second0..3).

All metadata files are hashed and copied to local source pins. Preparation reuse rejects changed acceptance evidence or changed source files. All seeds and source frame digests are disjoint across train/validation. Holdout cells remain(1,1),(2,3) for every source, including paired seeds.

## Recovery weighting

Both old corrections and new actual recoveries occupy the same25% correction pool;400 basic episodes occupy75%. Recovery anchors receive3x, transition-only anchors1.5x, ordinary anchors1x, with **max** overlap. Only eligible training anchors have nonzero probability.

Old `recovery_stages` retain their recorded stage names and starts. Ends extend through the next recorded close intent to the first expert frame with `physical_grasp_ok`, plus0.3s. Regrasp also includes `retry_opened` through stable grasp+0.3s. Transport/placement include `loaded_recovery_start` through `loaded_recovery_complete`. Expert masks exclude intervention gaps. Old close_transition corrections have no explicit recovery stages and retain transition1.5x; no recovery phase is invented. New episodes use only explicit expert recovery intervals.

The read-only old500 audit (`old500_recovery_audit.json`) gives113168 train windows,11961 validation windows,450/50 episode split and2397 excluded nonexpert frames. Old correction recovery anchors are5941/17360 (34.22% before weighting), receiving59.234% of the correction pool after3x normalization, or14.809% global mass. The final accepted dataset recomputes and reports actual normalized masses in `sampling_audit.json`, including disjoint recovery/transition-only/ordinary counts and masses. Per-kind interval union audit is in `old500_recovery_by_kind.json`; overlapping stage intervals must not be summed as distinct durations.

## Safe cache reuse

Old500 record ordering and lowdim source are preserved; normalization is recomputed using training expert frames from all accepted sources. Existing1500 crop arrays are immutable symlinks to the prior cache. Their full original digest verification is inherited only while cache/audit metadata and shapes are unchanged. Original JPEG provenance is checked against timestamps at first decode, with byte hashing if a source timestamp changed. New images alone are decoded through the exact old float32 letterbox/crop and224 resize. New crop hashes, original JPEG hashes, horizon masks, source masses and event-adjacent raw/cache equality are verified before GPU access.

## Commands after independent completion verification

Run these from this directory. The default command performs CPU preparation and cache verification only. No background watcher is running.

```bash
./run_training.sh --prepare-only   # accepted source audit, manifest, stats and masks
./run_training.sh --cache-only     # same gate, reuse500 and decode/verify new only
./run_training.sh --smoke          # explicit GPU4-update batch64 smoke, then reload check
./run_training.sh --train          # same smoke/reload gate, then fresh90k training
```

Settings: ACT chunk30, batch64,90000 optimizer updates, FP32, LR/backbone LR1e-5, weight decay1e-4, ImageNet ResNet18 backbone with a freshly initialized ACT policy, seed20261027. Output checkpoints are written here under `output/`; old runs remain intact. Smoke refuses peak allocation above20GiB. The saved policy includes the exact deployed image processor and trained normalization. Reload checks real observations from basic, old correction, and new recovery sources, with heldout examples preferred when available.

The launcher pins pipeline/runtime/backbone hashes. Once pinned, code or environment changes require a new run directory. It refuses existing model output directories. Final checkpoint reload is mandatory after the90k updates.

CPU validation: `python -m unittest discover -s tests -v` (use the workspace `.venv/bin/python`). Tests cover horizon30 boundaries, masks, max weights and source mass, heldout seed/layout leakage, old/new count/acceptance, duplicates, real dataset action reads, frozen metadata changes, and inherited raw image changes. End-to-end new cache, smoke and reload validation remain gated on accepted collection completion.
