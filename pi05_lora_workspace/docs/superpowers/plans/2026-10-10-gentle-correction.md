# Gentle correction LoRA implementation plan
Goal: launch a verified new LoRA run from the 80000-microstep adapter with original/correction sampling mass 70/30 and 2x final-approach/recovery window weighting.
Architecture: isolated preparation, sampling and training scripts under gentle_correction_20261010; reuse existing LeRobot conversion, aggregation and trainer APIs without modifying shared library code.
1. Stop collection, snapshot accepted episodes and hashes. Convert continuous raw gripper commands with the existing phase-based binary labeling. Preserve all frames and all three cameras.
2. Aggregate original 160 episodes and 32 accepted corrections. Hold out 4 correction episodes by category; original episodes are all previously seen by initial checkpoint and must not be described as held-out validation.
3. Define final approach as open-intent anchors within 2s of close onset; union with logged correction recovery intervals. Normalize 2x/1x within original and correction pools, then assign masses .7/.3. Keep chunks chronological and within episode via existing dataset loader.
4. Test pool masses, ratio, holdout exclusion and deterministic sampler resume. Verify converted actions/states, video decoding, processor normalization, trainable modules and loaded adapter.
5. Fresh optimizer/scheduler, LR 1e-5 to 1e-6, 50-update warmup, 2000 optimizer updates; microbatch1 x accumulation8, chunk15. Save/evaluate every250 optimizer updates; additionally save after first update for verification.
6. Start user service after confirming free GPU. Verify actual optimizer update, finite loss and checkpoint. Document exact inputs and commands.

## User revision: base initialization and 80000 microsteps
User superseded continuation initialization: stop previous service, retain its outputs, initialize fresh LoRA from pi05_base in isolated gentle_correction_base_20261010. Reuse frozen data/split/70:30/2x sampler. Peak LR5e-5/end5e-6, warmup4000 microsteps (500 updates), budget80000 microsteps (10000 updates); save and evaluate every4000 microsteps (500 updates). Verify all252 LoRA B matrices zero and projection/time MLP modules unchanged from loaded base before the first update; verify first update changes adapter tensors and finite checkpoint saves.
