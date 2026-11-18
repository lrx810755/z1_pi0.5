# Cup shape / height comparison Implementation Plan

Goal: 120 independent full-robot grasp and proof-lift trials, two collision shapes, two heights, three fixed grid centres, ten repeats each.
Architecture: experiment-only snapshot of current horizontal assistant and scene, private launch and configs. Restart isolated ROS domain for every shape/height group. Run current production plan/execute/guided-grasp services, stop before box transport. Preserve all traces and camera videos; report planning, approach, pinch/lift and physical success separately.
Constraints: frustum Rbottom18mm/Rtop25.2mm/H61mm; old cylinder R25.2mm/H61mm; height34/32mm; unchanged80g, friction, PD, speed, thresholds, route order. Near/mid/far grid columns0/2/4 row1, exact centres derived from current workspace. Keep original scanned appearance to avoid additional visual changes. Physics collider and MoveIt must match. No collection/training resume. Headed mode, three cameras10Hz, dedicated domain135/port8838.
Tasks:
- [ ] Pure case matrix and outcome classification: tests first, verify120 unique cases, paired coordinates, reject service-only false positives.
- [ ] Private snapshots, same-shape physical and MoveIt adapters, group-specific config (height only), fixed-pose randomizer. Verify geometry payload and unchanged source hashes.
- [ ] Reuse production experiment probe with isolated endpoint/domain; add column setter and actual planning-scene verification; preserve every result even on failures.
- [ ] Background orchestration of four groups, bounded readiness/cleanup, resumable without overwriting; aggregate final Markdown/JSON including complete counts and failures.
Review focus: false success from service responses; pose changes between repeats; MoveIt collider differs from PhysX; partial runs presented as completed; timed-out service still executing at next reset.
