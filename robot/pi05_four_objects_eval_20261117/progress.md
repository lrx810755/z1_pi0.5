# SDD ledger — plan: docs/superpowers/plans/2026-11-17-four-object-eval.md

Ruling: 原工作目录不是Git仓库；使用独立新增目录，不建立或修改共享仓库。
Ruling: 用户已要求准备接口，且交接要求直接行动；按已有10Hz/15步契约实现，不增加审批停顿。
Pre-flight: Task 2 consumes Task 1 contracts; Task 3 tests both with real inputs.

Task 1: complete — final contract, seed/coverage and negative gates tested.
Task 2: complete — isolated snapshots, real parser/transport tests, manual SH entries and launch introspection pass.
Task 3: complete — 32 tests pass; 8 real raw-input CPU processing cases pass; README and verification.json saved.
Review: independent read-only reviewer identified processor JSON drift and stale weight verification; both fixed with RED/GREEN regressions. Base SHA/stat identity and frozen source-summary hashes also bound.
Ruling: No Git repository here; no commits or worktree integration. Only new isolated directory and three SH entries created.
Ruling: Native GPU verification deliberately pending until final model exists and GPU is accessible; no fake success markers, no evaluation launched.

2026-11-17 actual GPU launch: final100000 updates completed; strict final model reload and eight real raw RGB bridge predictions pass. Fixed verification-only action Tensor serialization and conda PYTHONNOUSERSITE inheritance with regression tests. 34 tests pass. Four objects x50 lifecycle job at /home/lp531/smolvla_eval/pi05_final_four_objects_50each_20261117_145347/job. No continuous monitoring service.

2026-11-17: User stopped execute15 evaluation at183 results and explicitly confirmed new execution-window comparison: predict30, execute25 and20;4objects x25 each group,total200. Added execute window selector and actual provenance,35 tests pass and GPU raw bridge8 pools pass with renewed source pins. Background comparison job: /home/lp531/smolvla_eval/pi05_final_execute25_20_25each_20261117_173702/job

2026-11-17: User confirmed40 trials,all4 objects10 each,predict30/execute30. Added30 selector,35 tests pass;8 pool GPU bridge verification passed. Background job /home/lp531/smolvla_eval/pi05_final_execute30_10each_20261117_203311/job

2026-11-17: User requested execution10,prediction30,40 trials=4objects x10 each. Added10 selector;35 tests passed and8 pool GPU bridge verification passed. Background job /home/lp531/smolvla_eval/pi05_final_execute10_10each_20261117_212120/job
