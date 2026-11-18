# Execution record
- User approved frustum and original cylinder; paired 34mm and32mm heights, near/middle/far10 each=120.
- Isolated snapshots and private adapters; no production files edited, original source hashes recorded.
- Matrix/geometry/physical acceptance tests RED→GREEN, then fixed-pose sampler regression RED→GREEN after missing pathlib import surfaced during first smoke run.
- Initial reset-only setup failures quarantined outside trial cases; no robot trial counted from that batch.
- Independent review: cylinder/mass/material ordering and height calibration verified. Important reset accounting/operation timeout findings fixed with valid_trial gate, quarantine, full scene restart, bounded3 infrastructure retries; test RED→GREEN. No repeated review requested.
- Actual near pose verified(.393,-.141,0), current frustum planning failures preserved as genuine planning outcomes.
- Ruling: inspection of the current production service shows its guided_grasp already includes carry to the box overhead, unlike the older Oct26 implementation. Preserve this behavior as requested, do not call guided_place, and report pinch/lift/transport failures distinctly. The final production-service pass rate includes this loaded carry.
- Ruling: preserve original visual scan while changing only physics and MoveIt collision geometry; no policy images used in this scripted test, eliminating appearance as a confound.
- Configured34/32mm are calibrated target heights, not measured contact heights; raw contact evidence retained. Report distinguishes trials that never reach closure.

- Preflight layout correction: original apple(.44,-.18) and peach(.54,-.16) obstructed near/middle fixed cup points. Original10 planning-only trials archived outside final120. All distractors fixed at y=.12, x=.43/.51/.60/.69 for every shape/height/position; control settings unchanged. First restarted near case planned/executed and passed original physical guided_grasp.
