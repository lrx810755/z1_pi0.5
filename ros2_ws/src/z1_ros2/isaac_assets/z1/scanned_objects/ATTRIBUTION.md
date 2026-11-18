# High-quality tabletop asset attribution

These visual assets were selected from the
[`barszot/3d-models-for-isaac-sim-dataset`](https://huggingface.co/datasets/barszot/3d-models-for-isaac-sim-dataset)
collection. The dataset is marked `cc-by-4.0`; its upstream `credits.csv`
lists each selected model as `CC Attribution`.

| Local asset | Original title | Creator | Source |
| --- | --- | --- | --- |
| `cup.usdz` | Indian Tea Glass (Chai) | bhagathgoud | [Sketchfab](https://sketchfab.com/3d-models/indian-tea-glass-chai-bc87e1579d544bbd9f7d132095e40e3d) |
| `apple.usdz` | Apple | Moshe Caine | [Sketchfab](https://sketchfab.com/3d-models/apple-da0cfd4c6a4e41c5aaf06797b463235b) |
| `peach.usdz` | Peach Polycam | Moshe Caine | [Sketchfab](https://sketchfab.com/3d-models/peach-polycam-b6a1d0ee969e4cbd921249be04b16e65) |
| `mouse.usdz` | Computer Mouse | kubassa | [Sketchfab](https://sketchfab.com/3d-models/computer-mouse-4a6e2b1cf5d54ed5b97da093ed1e74a4) |
| `beverage_bottle.usdz` | Coca-Cola Bottle | MemoRorschach | [Sketchfab](https://sketchfab.com/3d-models/coca-cola-bottle-34075fedb0ef40d9a172231134849914) |

The downloaded files are stored in `source/`. Normalized local USD wrappers
are stored in `usd/`. `manifest.json` records SHA-256 hashes, normalized
bounds, scale factors, collision approximations, and the number of source
physics APIs suppressed by each wrapper.

Preserve this file, `UPSTREAM_CREDITS.csv`, and `manifest.json` whenever the
visual assets are redistributed.

The previously selected coffee mug was rejected as too wide for the Z1
gripper and retained in `retired/cup_mug_ungraspable.usdz` for traceability.
