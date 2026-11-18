# Forward bottle loaded transport

Scope: new private bottle interface only; preserve old bottle entry, cup files, grasp calibration and40-success dataset. Restart only after verifying code. Previous uncontrolled scheduler is paused; preserve all records.
- [x] Pure guards: rigid-grasp target conversion with desired orientation,3mm maximum reverse drawdown,3mm vertical-lift XY and horizontal height corridor,20mm permitted lateral detour,0.035rad orientation bound. Reject nonfinite data.
- [x] New mixin: vertical lift first, fixed-orientation forward carry second; explicit Pilz LIN then audited OMPL fallback. Every candidate audited densely before execution. Reduce speed/acceleration for a LIN dynamics rejection; do not relax direction/orientation.
- [x] During loaded execution: monitor real bottle progress, FK attitude, retained relative pose40mm, feedback freshness, dual-contact loss with existing0.5s grace; cancel action on violation. Keep existing post-motion physical validation.
- [x] Tests cover target rotation, reverse reproduction, intermediate reversal, bounds, rejected plan never executed, cancellation on violation. Run complete private suite.
- [x] Reload private scene/collector with remaining quota; verify live request type, guard audit and first attempts. Do not promise completion/success from unit tests.

Live verification: ready at20:44:52; first lift LIN candidate audited240samples, predicted endpoint error1.38e-8m. Actual lift lateral deviation4.32mm triggered cancellation and failed-record archive; no completion/success-rate claim. Same40-success collection resumed.
