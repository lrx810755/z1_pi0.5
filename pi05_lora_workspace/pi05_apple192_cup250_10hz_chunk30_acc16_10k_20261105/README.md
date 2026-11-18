# Apple192 + cup250 joint LoRA pilot

User authorized fresh official PI0.5 base LoRA, rank32 alpha32 dropout0, 10Hz H30, execute default15, batch1 accumulation16, 10000 optimizer updates=160000microsteps; AdamW5e-5→5e-6 cosine, warmup500 updates=8000microsteps, decay160000; save/validation1000updates. BF16, gradient checkpointing, vision frozen. Same task-balanced adapter for both objects; source recordings remain unchanged.

192 apple (133 full +59 lift-only correction), 250 cup (200 basic,50 full correction). Basic cup10/cell; corrections50 stratified across20cells(2–3/cell) and correction category(half original count),5heldout. Preserve existing per-source spatial splits:395train47validation; object grid coverage differs. Source seed overlap across train/val forbidden. Partial apple corrections use task `Physically grasp the apple and lift it clear of the table.`; full episodes retain pick/place task. No expert/nonexpert window mixing, no cross-episode windows, raw RGB decode and full JPEG SHA256 pins, train-only shared quantile stats.

Sampling mass: apple full35%,apple correction15%,cup basic37.5%,cup correction12.5%; transition vicinity±0.8s multipliers1.5 inside pools. Thus objects50/50, source proportions70/30 and75/25. Do not infer shared target accuracy from loss alone.

Preparation/processor checks CPU now. pipeline waits for both24kcheckpoint evaluations execute30/40(each30) + queue completion + evaluation unit exit, then >=18000MiB free; only then GPU smoke/reload and fresh formal training. GPU/data/init failures stop rather than continue with wrong settings. Final checkpoint reload verified after completion. No periodic simulator evaluations scheduled inside training; saved1k checkpoints support later5k/10k comparisons. Training plan10k is fixed per user.
