# Source takeover candidate audit

19 distinct failed rollout episodes selected: 6 current 8 Hz + 13 old 10 Hz, representing 18 seeds. These are trace-screened takeover candidates; physical recovery success still requires replay and teacher execution. No source files were changed and no GPU/Isaac process was launched.

17 candidates have at least 15 mm horizontal policy displacement; two supplements have 10.6 mm (current episode 26) and 13.0 mm (old episode 13). Do not describe all 19 as >=15 mm. Current episode 11 was excluded because final_box_results.json counts it as a successful source rollout, even though its legacy result.json success is false.

Every accepted takeover has an uninterrupted >=0.4 s window with speed <=0.02 m/s, valid cylinder_tilt_deg <=10 degrees, reported object_touching_table=true, cup height within 8 mm of its settled baseline, gripper commanded open and measured <=-0.75 rad, not grasped, unfrozen, and target collision enabled. Distinct object pose timestamps must span the window, so a stale pose cannot manufacture stability. Any earlier invalid/over-10-degree tilt, >25 mm vertical excursion, physical grasp or lift excludes the prefix. In the accepted data, the actual worst prefix tilt is only 3.656 degrees, and worst vertical excursion is 0.836 mm.

All first 1 mm policy movements begin after 13 seconds; cup drift in the first 2 seconds is <=2 micrometres. Initial randomization displacement is not counted as a push. All candidates except current episode 27 report finger/object contact before takeover. Episode 27 moves at 14.15 seconds with gripper/cup origin distance down to 0.148 m, near the configured 0.152 m finger insertion location, but lacks a reported finger contact; this should be visually or physically confirmed before claiming contact-proven push causation.

Takeover workspace screen was expanded to x=[0.32,0.72], y=[-0.30,0.12] metres. This screen is not an IK or collision-free proof. Current episode 27 (y=-0.273 m) and old episode 14 (x=0.699 m) fall just outside original evaluation sampling bounds and are explicitly flagged. All retained takeovers have real table contact and near-zero root z.

Some original policies tilt/drop/fly the cup later. This does not invalidate an earlier safe takeover. The first later prefix-safety violation time and original final state are stored for each candidate; execute the prefix only through takeover.

Quaternion convention: use contact_status.object_orientation_xyzw and cylinder_tilt_deg for world uprightness. The initial_scene_randomization.object_orientations_xyzw field is historically mislabeled: run_z1_ros2_scene.py saves raw Isaac get_world_pose quaternion in WXYZ order. Each candidate saves explicitly named root WXYZ and evaluator/world XYZW alternatives. Never classify raw root quaternion as world-up by assuming its label is correct.

candidates_manifest.json is authoritative. Each candidate JSON includes complete source paths, seed, source episode, index, exact simulation time, cup pose in both quaternion conventions, all joints, command, complete five-object initial layout, stability window, prefix maximum tilt/speed/z, motion-onset evidence, final-box failure record, and original command prefix path. Prefix command JSONL retains source simulation timestamps, measured joint state, policy action, execution target, actual command, action/substep index, phase, and collision/frozen flags. The full original trace/input_sync/prediction_chunks paths remain linked.

Verified with python3 -m unittest -v test_select_candidates.py: 9 tests passed, including direct re-check of all full source prefixes, unique source episodes, exclusion of successful source rollouts, real table contact, sustained motion thresholds, stale pose rejection, prior tilt/flight rejection, and exact prefix length/pose provenance.

| Group | Episode | Seed | Takeover s | XY mm | Prefix max tilt deg | Notes |
|---|---:|---:|---:|---:|---:|---|
| current8hz | 1 | 1612956893 | 24.4 | 16.4 | 1.466 |  |
| current8hz | 5 | 888220176 | 16.983 | 48.4 | 2.877 |  |
| current8hz | 22 | 1076225076 | 16.533 | 31.3 | 1.262 |  |
| current8hz | 23 | 1473018436 | 18.7 | 31.3 | 3.656 |  |
| current8hz | 26 | 16865083 | 20.8 | 10.6 | 3.414 | 10 mm supplement |
| current8hz | 27 | 2028347721 | 15.183 | 23.7 | 0.106 | outside initial sampling; contact confirmation needed |
| old10hz | 2 | 2081576163 | 29.167 | 53.2 | 1.685 |  |
| old10hz | 3 | 318573086 | 42.6 | 15.8 | 2.752 |  |
| old10hz | 4 | 712193919 | 26.15 | 29.8 | 2.139 |  |
| old10hz | 11 | 1909058445 | 31.65 | 18.4 | 0.719 |  |
| old10hz | 12 | 2049048119 | 33.983 | 25.7 | 1.115 |  |
| old10hz | 13 | 1217702027 | 41.833 | 13.0 | 2.345 | 10 mm supplement |
| old10hz | 14 | 908046731 | 22.5 | 33.7 | 1.909 | outside initial sampling |
| old10hz | 15 | 161048945 | 24.667 | 16.6 | 0.785 |  |
| old10hz | 16 | 6082909 | 43.133 | 15.5 | 2.08 |  |
| old10hz | 19 | 607990330 | 32.55 | 32.8 | 2.43 |  |
| old10hz | 22 | 1076225076 | 36.4 | 15.7 | 2.897 |  |
| old10hz | 29 | 1509058260 | 39.3 | 15.4 | 1.782 |  |
| old10hz | 30 | 1721670981 | 26.65 | 18.9 | 2.074 |  |
