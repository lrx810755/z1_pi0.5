# Execution ledger — 2026-10-29 PI0.5 cup recovery LoRA

Approved spec: user-confirmed parameter table; user explicitly requested launch without ongoing training monitoring.

Ruling: use a new isolated experiment directory outside Git — this workspace is a collection of experiment artifacts, not a Git repository; shared LeRobot and all old experiments remain unchanged.

Task1 complete: independently audited14 saved episodes;13 accepted, episode_000005 rejected because actual replay tilted before takeover. All accepted JPEGs fully decoded and hashed. Old500 source pins verified. Prepared513 episodes,461 training/52 validation,147603 frames,116873 eligible training windows. Training-only quantiles and expert masks verified. Original19-completion marker/count gates were not modified or forged.

Task2 prepared and CPU-verified: native dataset/sampler/config hooks, runner and shell launcher, strict official base load guard, serialized crop/normalization processors. 252 LoRA target modules,55249952 trainable parameters. 10 CPU regression/contract tests pass; real model/PEFT meta construction and three real batch processor roundtrips pass. Old cache/source JPEG crop equality passes. GPU smoke, reload inference, actual first update and formal training remain unverified.

Task3 attempted, blocked: `bash launch.sh` returned1. Torch2.9.1+cu128 reports CUDA unavailable, /dev/nvidia* empty. Separate user-service probe returned operation not permitted. `launch_preflight.json` and `workflow_status.json` record training_started=false, stage=blocked. No formal output directory or first-update marker exists. No ongoing monitoring was started. Host launch command is documented in README.md.

Independent review: review_pi05_launch found native PI05 silent base-load fallback and nonfinite validation log parsing gaps. Both fixed locally, regression tests watched fail then pass. Follow-up reviewer confirmed both resolved. No shared LeRobot edits. GPU runtime remains unverified.

Final verification: unittest discovery10/10 OK, bash -n launch.sh run_training.sh OK, py_compile new Python files OK, cpu_verification.json passed=true. Actual launch failed for unavailable CUDA; do not report training running.

Resumed2026-10-29 16:07 with host GPU access. RTX4090 available; systemd service launched. CPU verification, real8-microstep smoke and two heldout checkpoint reload predictions passed. Formal run initialized fresh base/LoRA; first actual optimizer update verified, then progressed beyond10/30000 updates. launch_verified.json records active service and first-update evidence. Startup verification finished; no ongoing monitoring.
