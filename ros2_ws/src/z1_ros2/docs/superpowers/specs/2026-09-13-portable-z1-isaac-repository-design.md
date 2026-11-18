# Portable Z1 Isaac Repository Design

## Goal

Publish the current Z1 ROS 2 and Isaac Sim work to
`git@github.com:lrx810755/z1_ros2.git` as a reproducible repository for a new
RTX 5070 Ti computer. The primary migration target is Windows 11 with Ubuntu
22.04 in WSL2, ROS 2 Humble in WSL2, and Isaac Sim 5.1 installed separately.

The repository must preserve the user's current uncommitted ROS changes and
the active Isaac scenes, scripts, robot meshes, calibration, and tabletop
objects without committing machine caches, generated build trees, training
datasets, credentials, or the Isaac Sim installation itself.

## Current State

- The Git repository is `/home/lp531/ros2_ws/src/z1_ros2`.
- Its current `origin` is the upstream
  `https://github.com/idra-lab/z1_ros2.git`.
- The user-owned destination is
  `git@github.com:lrx810755/z1_ros2.git`.
- `/home/lp531/isaac_assets/z1` is not a usable Git repository, but the ROS
  launch files and tests refer to it directly.
- The active Isaac entry point is
  `isaac_assets/z1/scripts/run_z1_ros2_scene.py` after import.
- The active three-camera scene is
  `isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd` after
  import.
- The current workstation has no GitHub SSH key accepted by GitHub. Upload
  therefore requires browser-based GitHub CLI authentication or a newly
  installed SSH public key. A token exposed in chat must not be used.

## Repository Layout

Keep the existing ROS packages at the repository root and add the portable
assets beside them:

```text
z1_ros2/
├── isaac_assets/z1/
│   ├── configuration/
│   ├── generated/
│   ├── generated_dual_camera/
│   ├── meshes/
│   ├── scanned_objects/
│   ├── scripts/
│   ├── hand_eye_calibration.yaml
│   ├── z1.urdf
│   └── z1.usd
├── requirements/
│   ├── python.txt
│   ├── training.txt
│   └── yolo.txt
├── scripts/
│   ├── bootstrap_ubuntu_22_04.sh
│   ├── configure_environment.sh
│   ├── verify_reproduction.sh
│   └── start_ros_stack.sh
├── windows/
│   └── start_isaac_sim.ps1
├── z1_bringup/
├── z1_description/
├── z1_examples/
├── z1_hardware_interface/
├── z1_isaac_bridge/
├── z1_moveit/
├── .gitattributes
├── .gitignore
├── REPRODUCE.md
└── requirements.txt
```

The repository itself is cloned at `~/ros2_ws/src/z1_ros2` in WSL2. The
bootstrap script creates that workspace layout when necessary rather than
storing an extra `ros2_ws/src` directory inside Git.

## Asset Import Policy

Import the files required to open the current scenes and regenerate them:

- active `configuration`, `generated`, and `generated_dual_camera` USD/URDF
  files;
- Z1 collision and visual meshes;
- active source and wrapper assets for cup, apple, peach, mouse, and beverage
  bottle;
- `ATTRIBUTION.md`, `UPSTREAM_CREDITS.csv`, and `manifest.json` for asset
  licensing and integrity;
- active Isaac scene, validation, exporter, and teleoperation scripts;
- the current hand-eye calibration file.

Exclude the following:

- `.git`, `.codegraph`, `scripts/codegraph`, `__pycache__`, and `*.pyc`;
- `build`, `install`, `log`, `.pytest_cache`, IDE files, and runtime logs;
- files with names containing `backup`, `before_`, or ending in `.bak`;
- retired and candidate object assets not referenced by the active scene;
- generated preview images and historical work logs that are not needed for
  execution;
- `/home/lp531/isaac-sim` and all NVIDIA binaries;
- `/home/lp531/smolvla_data`, checkpoints, recordings, and other datasets;
- credentials, access tokens, private keys, and local Git credential stores.

Use Git LFS for `*.usd`, `*.usdz`, `*.dae`, `*.stl`, `*.STL`, and other large
binary model formats. Text source, configuration, attribution, and scripts
remain normal Git objects.

## Portable Paths

Remove executable defaults tied to `/home/lp531`.

- `Z1_REPO_ROOT` identifies the clone root.
- `Z1_ASSET_ROOT` defaults to `$Z1_REPO_ROOT/isaac_assets/z1`.
- `ISAAC_SIM_ROOT` identifies the separately installed Isaac Sim directory.
- `Z1_DATASET_ROOT` identifies a user-owned dataset directory outside Git.
- `Z1_REAL_ARM_CONTROLLER_DIR` and `Z1_REAL_ARM_SDK_DIR` remain optional for
  real-hardware operation.

Launch arguments continue to override all environment defaults. Historical
documents may describe old absolute paths, but executable Python and launch
files, active README commands, and tests must use portable paths.

Generated USD and JSON files that embed old absolute paths must either be
rewritten safely or regenerated from the checked-in source. The verification
script must fail if an active runtime file still contains `/home/lp531`.

## Dependency Manifests

Dependencies are divided by their installation owner:

- ROS package dependencies stay in `package.xml` and are installed with
  `rosdep` against ROS 2 Humble.
- `requirements/python.txt` contains non-ROS Python packages used by the ROS
  and utility scripts, with versions compatible with Python 3.10.
- `requirements/yolo.txt` preserves the tested Ultralytics stack.
- `requirements/training.txt` records the external LeRobot/training
  dependencies separately so simulation-only users do not install the full
  training stack.
- Root `requirements.txt` is a documented convenience entry point for the
  ordinary Python dependencies; it must not attempt to pip-install ROS 2 or
  Isaac Sim.
- `REPRODUCE.md` records Ubuntu 22.04, ROS 2 Humble, Isaac Sim 5.1, Fast DDS,
  Git LFS, NVIDIA driver expectations, and the WSL2 split-process limitation.

Isaac modules such as `isaacsim`, `omni`, `pxr`, `carb`, and `usdrt` are
provided by Isaac Sim and must not appear as PyPI requirements.

## WSL2 Runtime Model

The portable configuration supports two modes:

1. Native Ubuntu 22.04: the existing combined ROS launch may start Linux
   Isaac Sim when `start_isaac:=true`.
2. Windows 11 plus WSL2: `windows/start_isaac_sim.ps1` starts Windows-native
   Isaac Sim, while `scripts/start_ros_stack.sh` starts ROS 2, MoveIt, and the
   recorder in WSL2 with `start_isaac:=false`.

The WSL2 mode uses Fast DDS UDP between Windows and WSL2. It must not depend
on shared-memory transport across that boundary. The documentation must mark
three raw RGB streams as a throughput-sensitive path that requires validation
on the target computer.

## Preservation and Git History

Do not overwrite or discard the current dirty working tree. Add a new remote
named `personal` for the destination and preserve `origin` as the upstream
source. Stage files deliberately, excluding all generated build trees and
caches. Commit the current project changes and portability work in reviewable
commits before pushing `main` to `personal/main`.

Because the destination may already contain commits, fetch it before pushing.
If its history conflicts with the local upstream history, do not force-push;
inspect the destination and integrate or use a new migration branch.

## Verification

Before upload:

1. Scan tracked candidates for tokens, private keys, and common secret names.
2. Confirm no ordinary Git object exceeds GitHub's 100 MiB limit and all
   configured binary asset patterns are handled by Git LFS.
3. Validate `scanned_objects/manifest.json` SHA-256 values for the five active
   source assets.
4. Run Python syntax compilation for imported scripts without importing Isaac
   modules.
5. Run the existing unit tests that do not require a live Isaac session.
6. Build the ROS workspace with `colcon build --symlink-install`.
7. Run a path audit that rejects `/home/lp531` in active executable files.
8. Create a temporary local clone and run the repository verification script
   from that clone.
9. Inspect the exact staged file list and total Git/LFS size.
10. Push without force and verify the remote branch and commit ID.

Full Windows/WSL2 GPU rendering cannot be proven on the current Ubuntu host.
The repository can verify configuration and document a target-machine smoke
test, but final camera throughput and DDS behavior must be measured on the
5070 Ti computer.
