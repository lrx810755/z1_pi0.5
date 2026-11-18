# Independent bottle horizontal supplement interface

Goal: prepare a new headed10Hz three-camera bottle demonstration entry without changing old bottle/cup interfaces or starting a simulator.
Design: private copy of current cup horizontal route code, frozen production grasp assistant, private launch/config/recorder/column setter. Target beverage_bottle only; ROS_DOMAIN_ID136, HTTP8839, separate data/log/failure directories. The first column uses no retreat; other columns exhaust low50/60/80/100mm, then high80/120mm, then direct fallback. One blended trajectory, final scaling.064. Keep bottle calibration68mm, pitch−5..−8°, preclose bands, preload5°, displacement8mm, contact ratio.35, transport/placement unchanged. Cup-only25mm/10mm/direct-release changes do not become bottle settings.
Tasks:
- [x] Add failing interface tests for isolated paths/target, bottle calibration, quota validation, preview side effects, bounded column setting.
- [x] Create private snapshot and adapt only target/entry paths/domain/port; bottle pitch ceiling8° preserves bottle candidates.
- [x] Add configurable one-command launcher(default10 successes/cell=200), manifest, README. Preview performs no filesystem mutation; live checks GPU/shared scene lock and preserves existing samples.
- [x] Run existing route/unit tests, new entry tests, ROS launch argument parsing, fresh bottle profile import and constructor check in isolated ROS domain; verify original source hashes unchanged. No simulator or recorder launch.
