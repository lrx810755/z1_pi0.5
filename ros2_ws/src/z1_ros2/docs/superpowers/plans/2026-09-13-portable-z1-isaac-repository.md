# Portable Z1 Isaac Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the current Z1 ROS 2 code and its active Isaac assets as a reproducible repository for an RTX 5070 Ti computer, including a documented Windows 11 plus WSL2 deployment path.

**Architecture:** Keep the existing ROS packages at the repository root, import the active Isaac asset set under `isaac_assets/z1`, and resolve all runtime locations through shared environment variables. ROS and ordinary Python dependencies are installed separately from Isaac Sim; Windows runs Isaac while WSL2 runs ROS nodes in the split deployment.

**Tech Stack:** Ubuntu 22.04, Windows 11/WSL2, ROS 2 Humble, MoveIt 2, Isaac Sim 5.1, Python 3.10/3.11, Fast DDS, Git LFS, PowerShell

**Spec:** `docs/superpowers/specs/2026-09-13-portable-z1-isaac-repository-design.md`

## Global Constraints

- Destination repository: `git@github.com:lrx810755/z1_ros2.git`.
- Preserve the current dirty working tree; never discard or overwrite user changes.
- Do not use the access token exposed in chat and do not place credentials in files, Git configuration URLs, or shell history.
- Do not commit `/home/lp531/isaac-sim`, `/home/lp531/smolvla_data`, build trees, caches, CodeGraph data, backups, or runtime logs.
- Preserve `scanned_objects/ATTRIBUTION.md`, `UPSTREAM_CREDITS.csv`, and `manifest.json` with redistributed object assets.
- Use Git LFS for binary USD, USDZ, DAE, and STL assets.
- ROS 2 and Isaac Sim are not PyPI requirements.
- Native Ubuntu remains supported; Windows 11 plus WSL2 uses Windows-native Isaac and WSL2-native ROS 2.
- Never force-push the destination repository.

---

### Task 1: Protect the working tree and import the active asset set

**Files:**
- Modify: `.gitignore`
- Create: `.gitattributes`
- Create: `isaac_assets/z1/**`
- Create: `z1_isaac_bridge/test/test_repository_assets.py`
- Modify: current source, launch, configuration, test, and documentation files already present in the dirty working tree.

**Interfaces:**
- Consumes: `/home/lp531/isaac_assets/z1` as the read-only asset source.
- Produces: `isaac_assets/z1` as the repository-owned asset root and Git LFS rules for binary models.

- [ ] **Step 1: Extend the exclusion rules before staging anything**

Add these exact repository-level rules:

```gitignore
build/
install/
log/
**/__pycache__/
*.py[cod]
.pytest_cache/
.codegraph/
**/.codegraph/
isaac_assets/z1/scripts/codegraph/
*.bak
*backup*
*before_*
smolvla_data/
checkpoints/
z1_bringup/launch/helloworld.cpp
```

- [ ] **Step 2: Preserve the pre-existing project work in its own commit**

Review every remaining modified or untracked path. Stage only source,
configuration, tests, and documentation from `docs`, `z1_bringup`,
`z1_description`, `z1_examples`, `z1_isaac_bridge`, and `z1_moveit`. Confirm
that ignored caches, build trees, backups, and `helloworld.cpp` are absent,
then run the current non-GPU tests and ROS build:

```bash
git add .gitignore docs z1_bringup z1_description z1_examples z1_isaac_bridge z1_moveit
git diff --cached --check
python3 -m pytest z1_isaac_bridge/test z1_examples/test -q
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
cd /home/lp531/ros2_ws/src/z1_ros2
git commit -m "feat: add Z1 Isaac training and grasp workflow"
```

- [ ] **Step 3: Write an asset contract test and verify that it fails**

Create `z1_isaac_bridge/test/test_repository_assets.py`:

```python
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
ASSETS = REPOSITORY / "isaac_assets" / "z1"


def test_active_asset_bundle_is_complete():
    required = (
        "generated_dual_camera/smolvla_dual_wrist_camera.usd",
        "generated_dual_camera/z1_with_dual_d435.usd",
        "scripts/run_z1_ros2_scene.py",
        "scripts/layout_sweep.py",
        "scanned_objects/source/apple.usdz",
        "scanned_objects/source/beverage_bottle.usdz",
        "scanned_objects/source/cup.usdz",
        "scanned_objects/source/mouse.usdz",
        "scanned_objects/source/peach.usdz",
        "scanned_objects/ATTRIBUTION.md",
        "scanned_objects/UPSTREAM_CREDITS.csv",
        "scanned_objects/manifest.json",
    )
    missing = [name for name in required if not (ASSETS / name).is_file()]
    assert not missing, missing


def test_asset_bundle_contains_no_machine_artifacts():
    forbidden = {".git", ".codegraph", "__pycache__", "codegraph"}
    offenders = [path for path in ASSETS.rglob("*") if path.name in forbidden]
    offenders += [
        path for path in ASSETS.rglob("*")
        if path.is_file()
        and ("backup" in path.name.lower() or "before_" in path.name or path.suffix == ".bak")
    ]
    assert not offenders, offenders
```

Run:

```bash
python3 -m pytest z1_isaac_bridge/test/test_repository_assets.py -q
```

Expected: FAIL because `isaac_assets/z1` does not yet exist inside the repository.

- [ ] **Step 4: Import the curated files without changing the source directory**

Create the destination and copy these exact groups with `rsync -a`:

```text
configuration/
generated/configuration/
generated/smolvla_red_ball_into_box.json
generated/smolvla_red_ball_into_box.usd
generated/z1_with_d435.urdf
generated/z1_with_d435.usd
generated/z1_with_d435_resolved.urdf
generated_dual_camera/configuration/
generated_dual_camera/smolvla_dual_wrist_camera.json
generated_dual_camera/smolvla_dual_wrist_camera.usd
generated_dual_camera/z1_with_dual_d435.urdf
generated_dual_camera/z1_with_dual_d435.usd
generated_dual_camera/z1_with_dual_d435_resolved.urdf
meshes/
scanned_objects/source/
scanned_objects/usd/
scanned_objects/ATTRIBUTION.md
scanned_objects/GRASPABILITY_REPORT.md
scanned_objects/UPSTREAM_CREDITS.csv
scanned_objects/manifest.json
scripts/*.py
scripts/*.html
scripts/*.md
scripts/z1_teleop_backend/
hand_eye_calibration.yaml
z1.urdf
z1.usd
z1/configuration/
```

Do not copy candidate/retired objects, previews, work logs, nested `.git`
directories, or CodeGraph content.

Use an explicit source and destination and copy directories with cache
exclusions:

```bash
asset_source=/home/lp531/isaac_assets/z1
asset_destination=/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1
mkdir -p "$asset_destination"
rsync -a --exclude='.git/' --exclude='.codegraph/' --exclude='__pycache__/' \
  "$asset_source/configuration/" "$asset_destination/configuration/"
rsync -a --exclude='__pycache__/' \
  "$asset_source/meshes/" "$asset_destination/meshes/"
rsync -a --exclude='__pycache__/' --exclude='codegraph/' \
  "$asset_source/scripts/" "$asset_destination/scripts/"
mkdir -p "$asset_destination/scanned_objects"
rsync -a "$asset_source/scanned_objects/source/" \
  "$asset_destination/scanned_objects/source/"
rsync -a "$asset_source/scanned_objects/usd/" \
  "$asset_destination/scanned_objects/usd/"
for name in ATTRIBUTION.md GRASPABILITY_REPORT.md UPSTREAM_CREDITS.csv manifest.json; do
  install -m 0644 "$asset_source/scanned_objects/$name" \
    "$asset_destination/scanned_objects/$name"
done
for name in hand_eye_calibration.yaml z1.urdf z1.usd; do
  install -m 0644 "$asset_source/$name" "$asset_destination/$name"
done
rsync -a "$asset_source/z1/configuration/" \
  "$asset_destination/z1/configuration/"
```

After the directory copies, use `install -D -m 0644` for the enumerated
`generated/` and `generated_dual_camera/` files above. Remove any excluded
cache or backup found by the asset contract test before staging.

- [ ] **Step 5: Configure Git LFS before adding binary files**

Create `.gitattributes`:

```gitattributes
*.usd filter=lfs diff=lfs merge=lfs -text
*.usdz filter=lfs diff=lfs merge=lfs -text
*.dae filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.STL filter=lfs diff=lfs merge=lfs -text
```

Run:

```bash
git lfs install --local
git check-attr filter -- isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd
python3 -m pytest z1_isaac_bridge/test/test_repository_assets.py -q
```

Expected: the attribute is `lfs` and both tests PASS.

- [ ] **Step 6: Commit only Git LFS configuration and imported assets**

```bash
git add .gitignore .gitattributes isaac_assets/z1 z1_isaac_bridge/test/test_repository_assets.py
git diff --cached --check
git lfs ls-files
git commit -m "build: bundle active Isaac assets with Git LFS"
```

### Task 2: Centralize portable ROS-side paths

**Files:**
- Create: `z1_isaac_bridge/z1_isaac_bridge/paths.py`
- Create: `z1_isaac_bridge/test/test_portable_paths.py`
- Modify: `z1_isaac_bridge/launch/z1_isaac.launch.py`
- Modify: `z1_isaac_bridge/z1_isaac_bridge/episode_recorder.py`
- Modify: `z1_isaac_bridge/z1_isaac_bridge/servo_node_vendor.py`
- Modify: `z1_isaac_bridge/setup.py`

**Interfaces:**
- Consumes: `Z1_REPO_ROOT`, `Z1_ASSET_ROOT`, `ISAAC_SIM_ROOT`, `Z1_DATASET_ROOT`, and `Z1_SERVO_PREFIX`.
- Produces: `repository_root()`, `asset_root()`, `isaac_sim_root()`, `dataset_root(name)`, and `servo_prefix()` returning absolute `Path` objects.

- [ ] **Step 1: Write failing resolver tests**

Test explicit environment overrides and source-checkout fallbacks:

```python
def test_environment_overrides_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("Z1_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("Z1_ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("Z1_DATASET_ROOT", str(tmp_path / "data"))
    assert paths.repository_root() == tmp_path / "repo"
    assert paths.asset_root() == tmp_path / "assets"
    assert paths.dataset_root("episodes") == tmp_path / "data" / "episodes"


def test_checkout_asset_fallback_points_inside_repository(monkeypatch):
    monkeypatch.delenv("Z1_REPO_ROOT", raising=False)
    monkeypatch.delenv("Z1_ASSET_ROOT", raising=False)
    assert paths.asset_root() == REPOSITORY / "isaac_assets" / "z1"
```

Run the exact test file and confirm import failure before implementation.

- [ ] **Step 2: Implement the path resolver**

Use `Path.expanduser()` and `Path.resolve(strict=False)`. Dataset fallback is
`Path.home() / "smolvla_data" / name`; Isaac fallback is
`Path.home() / "isaac-sim"`; Servo fallback is `/opt/ros/humble`.

- [ ] **Step 3: Replace executable `/home/lp531` defaults**

Update the ROS launch, recorder, and Servo wrapper to call the resolver. Add
`z1_isaac_bridge.paths` to `setup.py` through the existing package discovery;
do not install assets as ROS package data.

- [ ] **Step 4: Run targeted tests and commit**

```bash
python3 -m pytest z1_isaac_bridge/test/test_portable_paths.py z1_isaac_bridge/test/test_episode_recorder.py -q
git add z1_isaac_bridge
git diff --cached --check
git commit -m "refactor: make ROS and dataset paths portable"
```

### Task 3: Make Isaac-side scripts portable on Linux and Windows

**Files:**
- Create: `isaac_assets/z1/scripts/z1_paths.py`
- Create: `z1_isaac_bridge/test/test_isaac_script_paths.py`
- Modify: `isaac_assets/z1/scripts/run_z1_ros2_scene.py`
- Modify: `isaac_assets/z1/scripts/prepare_scanned_assets.py`
- Modify: `isaac_assets/z1/scripts/validate_wrist_camera_pose.py`
- Modify: `isaac_assets/z1/scripts/export_z1_lerobot_dataset.py`
- Modify: `isaac_assets/z1/scripts/inspect_usd.py`
- Modify: `isaac_assets/z1/scripts/z1_teleop_backend/config.py`
- Modify: `isaac_assets/z1/generated/*.json`
- Modify: `isaac_assets/z1/generated_dual_camera/*.json`

**Interfaces:**
- Consumes: the same `Z1_*` environment contract as Task 2.
- Produces: asset-local `ROOT`, `SCENE`, `DATASET_ROOT`, and platform-aware ROS bridge library configuration.

- [ ] **Step 1: Write a failing source audit**

The test reads active `*.py`, `*.json`, and launch files and asserts that none
contains `/home/lp531`. It excludes Markdown history, generated URDF comments,
and binary USD files from text scanning.

- [ ] **Step 2: Add the asset-local resolver**

`z1_paths.py` derives the asset root from
`Path(__file__).resolve().parents[1]` unless `Z1_ASSET_ROOT` is set and derives
datasets from `Z1_DATASET_ROOT` or `Path.home() / "smolvla_data"`.

- [ ] **Step 3: Update active scripts and JSON metadata**

Replace hard-coded executable paths with the resolver. In
`run_z1_ros2_scene.py`, configure `LD_LIBRARY_PATH` only when `os.name ==
"posix"`; Windows uses Isaac's bundled environment without a Linux shared
library path.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m pytest z1_isaac_bridge/test/test_isaac_script_paths.py -q
python3 -m compileall -q isaac_assets/z1/scripts z1_isaac_bridge/z1_isaac_bridge
git add isaac_assets/z1 z1_isaac_bridge/test/test_isaac_script_paths.py
git diff --cached --check
git commit -m "refactor: make Isaac scripts location independent"
```

### Task 4: Record reproducible dependency groups

**Files:**
- Create: `requirements.txt`
- Create: `requirements/python.txt`
- Create: `requirements/yolo.txt`
- Create: `requirements/training.txt`
- Create: `requirements/README.md`
- Modify: `z1_isaac_bridge/package.xml`
- Modify: `z1_examples/package.xml`
- Create: `z1_isaac_bridge/test/test_dependency_manifests.py`

**Interfaces:**
- Consumes: imports audited from the checked-in Python sources.
- Produces: rosdep-owned dependencies and three opt-in pip dependency groups.

- [ ] **Step 1: Write failing manifest coverage tests**

Assert that root `requirements.txt` includes `-r requirements/python.txt`,
that no requirements file contains `rclpy`, `isaacsim`, `omni`, `pxr`,
`carb`, or `usdrt`, and that the training group contains `lerobot==0.3.4`.

- [ ] **Step 2: Create the dependency groups**

Use these tested compatibility constraints:

```text
# requirements/python.txt
numpy>=1.21.5,<2
PyYAML>=5.4.1,<7
scipy>=1.8,<2
PyQt5>=5.15,<6
psutil>=5.9,<7
```

```text
# requirements/yolo.txt
numpy<2
opencv-python-headless==4.11.0.86
matplotlib>=3.5,<4
pillow>=9,<12
pyyaml>=5.4.1,<7
requests>=2.25,<3
psutil>=5.9,<7
polars>=0.20
nvidia-ml-py>=12
ultralytics-thop>=2.0.18
ultralytics==8.4.95
```

```text
# requirements/training.txt
lerobot==0.3.4
```

Keep Torch installation out of the generic files because the CUDA wheel must
match the target driver and the chosen LeRobot environment.

- [ ] **Step 3: Complete ROS package dependency declarations**

Declare the ROS/Ubuntu owners for OpenCV, NumPy, SciPy, Qt, `cv_bridge`, and
the message packages actually imported. Do not replace apt ROS packages with
pip packages.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest z1_isaac_bridge/test/test_dependency_manifests.py -q
rosdep check --from-paths . --ignore-src
git add requirements.txt requirements z1_isaac_bridge/package.xml z1_examples/package.xml z1_isaac_bridge/test/test_dependency_manifests.py
git diff --cached --check
git commit -m "build: document ROS Python and training dependencies"
```

### Task 5: Add Ubuntu, WSL2, and Windows launch helpers

**Files:**
- Create: `scripts/configure_environment.sh`
- Create: `scripts/bootstrap_ubuntu_22_04.sh`
- Create: `scripts/start_ros_stack.sh`
- Create: `windows/start_isaac_sim.ps1`
- Create: `z1_isaac_bridge/config/fastdds_wsl2.xml`
- Create: `z1_isaac_bridge/test/test_platform_launchers.py`

**Interfaces:**
- Consumes: repository-relative paths and optional `ISAAC_SIM_ROOT`/`Z1_DATASET_ROOT` overrides.
- Produces: a sourced shell environment, idempotent Ubuntu dependency setup, WSL ROS launcher, Windows Isaac launcher, and UDP-only Fast DDS profile.

- [ ] **Step 1: Write failing launcher contract tests**

Parse both DDS XML files. Assert the native profile includes `SHM` and `UDPv4`
and the WSL profile includes `UDPv4` but not `SHM`. Assert the shell scripts
contain `set -euo pipefail`, and parse the PowerShell file with a delimiter and
brace balance smoke test on Linux.

- [ ] **Step 2: Implement the environment and bootstrap scripts**

`configure_environment.sh` exports:

```bash
export Z1_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export Z1_ASSET_ROOT="$Z1_REPO_ROOT/isaac_assets/z1"
export Z1_DATASET_ROOT="${Z1_DATASET_ROOT:-$HOME/smolvla_data}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/isaac-sim}"
export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
```

The bootstrap installs Git LFS and the ROS dependencies with `rosdep`, then
builds from the enclosing workspace with `colcon build --symlink-install`.

- [ ] **Step 3: Implement split WSL2 launchers**

`start_ros_stack.sh` sources ROS Humble and the workspace, selects
`fastdds_wsl2.xml`, and runs `z1_isaac.launch.py start_isaac:=false`.

`windows/start_isaac_sim.ps1` accepts mandatory `-IsaacSimRoot` and optional
`-RepoRoot`, sets the shared environment contract, and invokes:

```powershell
& (Join-Path $IsaacSimRoot "python.bat") `
  (Join-Path $env:Z1_ASSET_ROOT "scripts\run_z1_ros2_scene.py") `
  "--scene" `
  (Join-Path $env:Z1_ASSET_ROOT "generated_dual_camera\smolvla_dual_wrist_camera.usd")
```

- [ ] **Step 4: Test and commit**

```bash
python3 -m pytest z1_isaac_bridge/test/test_platform_launchers.py -q
bash -n scripts/configure_environment.sh scripts/bootstrap_ubuntu_22_04.sh scripts/start_ros_stack.sh
git add scripts windows z1_isaac_bridge/config/fastdds_wsl2.xml z1_isaac_bridge/test/test_platform_launchers.py
git diff --cached --check
git commit -m "feat: add native and WSL2 reproduction launchers"
```

### Task 6: Document and verify a clean clone

**Files:**
- Create: `REPRODUCE.md`
- Create: `scripts/verify_reproduction.sh`
- Modify: `README.md`
- Modify: `isaac_assets/z1/generated_dual_camera/README.md`
- Modify: `z1_isaac_bridge/README.md`

**Interfaces:**
- Consumes: all artifacts from Tasks 1-5.
- Produces: one authoritative installation guide and a read-only verification command.

- [ ] **Step 1: Write the verification script**

The script checks required files, Git LFS pointers, five manifest hashes,
Python syntax, XML well-formedness, shell syntax, forbidden active paths, and
secret filename/content patterns. It exits nonzero on the first failed group
and never launches the robot or Isaac Sim.

- [ ] **Step 2: Write the reproduction guide**

Document these ordered flows:

1. clone with Git LFS;
2. native Ubuntu 22.04 setup;
3. Windows 11 plus WSL2 split setup;
4. Isaac Sim 5.1 installation outside Git;
5. ROS dependency installation and workspace build;
6. simulation smoke test;
7. three-camera DDS throughput check;
8. dataset directory setup outside Git;
9. optional YOLO and LeRobot environments;
10. real Z1 and RealSense prerequisites, clearly separated from simulation.

- [ ] **Step 3: Update README entry points and run local verification**

```bash
bash scripts/verify_reproduction.sh
python3 -m pytest z1_isaac_bridge/test z1_examples/test -q
git add REPRODUCE.md README.md scripts/verify_reproduction.sh isaac_assets/z1/generated_dual_camera/README.md z1_isaac_bridge/README.md
git diff --cached --check
git commit -m "docs: add clean-machine reproduction workflow"
```

### Task 7: Run full verification and upload

**Files:**
- Modify: Git remote configuration only after authentication succeeds.
- Exclude: `.codegraph`, `build`, `install`, `log`, `__pycache__`, backup Xacro files, and `z1_bringup/launch/helloworld.cpp`.

**Interfaces:**
- Consumes: the complete portable repository and destination GitHub repository.
- Produces: verified local commits and `personal/main` on GitHub.

- [ ] **Step 1: Run the full repository evidence suite**

```bash
bash scripts/verify_reproduction.sh
python3 -m pytest z1_isaac_bridge/test z1_examples/test -q
git status --short
git lfs ls-files
git count-objects -vH
```

Expected: verification and tests exit 0; status contains no unintended
artifacts; all USD/USDZ/DAE/STL assets appear in the LFS list.

- [ ] **Step 2: Verify from a fresh local clone**

Clone the repository into a `mktemp -d` directory using `--local`, run
`git lfs checkout`, and execute `bash scripts/verify_reproduction.sh`. Keep the
temporary clone until its result is recorded, then remove only that exact
`mktemp` directory.

- [ ] **Step 3: Authenticate without the exposed token**

Install Ubuntu's GitHub CLI package only when the command is absent, then use
the browser device flow:

```bash
if ! command -v gh >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y gh
fi
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git --hostname github.com
gh auth status --hostname github.com
```

Let the user approve the device in their browser. Do not use `--show-token`,
do not echo credentials, and do not store a token in a remote URL.

- [ ] **Step 4: Inspect the destination and push without force**

```bash
git remote add personal https://github.com/lrx810755/z1_ros2.git
git fetch personal
git log --oneline --left-right --graph personal/main...main
git push -u personal main
git ls-remote personal refs/heads/main
```

If `personal` already exists, verify its URL instead of replacing it. If the
destination has unrelated commits, stop and integrate them; never use
`--force` or `--force-with-lease`.

- [ ] **Step 5: Report immutable upload evidence**

Report the local HEAD, remote `refs/heads/main` hash, Git LFS object count and
size, excluded directories, tests executed, ROS build result, and the target
machine smoke tests that remain unverified.
