#!/usr/bin/env bash
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
z1_repo_root="$(cd "$z1_script_dir/.." && pwd -P)"
cd "$z1_repo_root"

command -v git >/dev/null
git lfs env >/dev/null

z1_required_files=(
  ".gitattributes"
  "requirements.txt"
  "isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd"
  "isaac_assets/z1/generated_dual_camera/z1_with_dual_d435.usd"
  "isaac_assets/z1/scripts/run_z1_ros2_scene.py"
  "isaac_assets/z1/scanned_objects/ATTRIBUTION.md"
  "isaac_assets/z1/scanned_objects/UPSTREAM_CREDITS.csv"
  "isaac_assets/z1/scanned_objects/manifest.json"
  "z1_isaac_bridge/config/fastdds_camera.xml"
  "z1_isaac_bridge/config/fastdds_wsl2.xml"
)
for z1_file in "${z1_required_files[@]}"; do
  if [[ ! -s "$z1_file" ]]; then
    echo "Required file is missing or empty: $z1_file" >&2
    exit 1
  fi
done

z1_lfs_count=0
while IFS= read -r z1_file; do
  [[ -n "$z1_file" ]] || continue
  z1_lfs_count=$((z1_lfs_count + 1))
  if ! git show ":$z1_file" | head -n 1 | grep -q \
      '^version https://git-lfs.github.com/spec/v1$'; then
    echo "Git index does not contain an LFS pointer for: $z1_file" >&2
    exit 1
  fi
done < <(git lfs ls-files -n)
if (( z1_lfs_count == 0 )); then
  echo "No Git LFS objects are tracked." >&2
  exit 1
fi

while IFS= read -r z1_file; do
  [[ -n "$z1_file" ]] || continue
  if [[ "$(git check-attr filter -- "$z1_file")" != *": lfs" ]]; then
    echo "Binary asset is not covered by Git LFS: $z1_file" >&2
    exit 1
  fi
  if ! git show ":$z1_file" | head -n 1 | grep -q \
      '^version https://git-lfs.github.com/spec/v1$'; then
    echo "Binary asset is covered by LFS rules but stored as a Git blob: $z1_file" >&2
    exit 1
  fi
done < <(git ls-files '*.usd' '*.usdz' '*.dae' '*.stl' '*.STL')

# Verify all five redistributed source objects against manifest SHA256 values.
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("isaac_assets/z1")
manifest = json.loads((root / "scanned_objects/manifest.json").read_text())
assets = manifest.get("assets", [])
if len(assets) != 5:
    raise SystemExit(f"expected 5 manifested source assets, found {len(assets)}")
for item in assets:
    path = root / item["source_file"]
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != item["sha256"]:
        raise SystemExit(f"SHA256 mismatch: {path}")
PY

z1_verify_cache="$(mktemp -d)"
trap 'rm -rf -- "$z1_verify_cache"' EXIT
PYTHONPYCACHEPREFIX="$z1_verify_cache" python3 -m compileall -q \
  isaac_assets/z1/scripts \
  z1_isaac_bridge/z1_isaac_bridge \
  z1_examples/z1_examples

python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET

for path in Path("z1_isaac_bridge/config").glob("*.xml"):
    ET.parse(path)
PY

bash -n \
  scripts/configure_environment.sh \
  scripts/bootstrap_ubuntu_22_04.sh \
  scripts/start_ros_stack.sh \
  scripts/verify_reproduction.sh

python3 - <<'PY'
from pathlib import Path

root = Path.cwd()
assets = root / "isaac_assets/z1"
active = list((assets / "scripts").rglob("*.py"))
active += list((assets / "generated").glob("*.json"))
active += list((assets / "generated_dual_camera").glob("*.json"))
active += list((root / "z1_isaac_bridge/launch").glob("*.py"))
offenders = [str(path.relative_to(root)) for path in active
             if "/home/lp531" in path.read_text(encoding="utf-8")]
if offenders:
    raise SystemExit("machine-specific active paths: " + ", ".join(offenders))
PY

if git grep -q -I -E \
    'github_pat_|ghp_[[:alnum:]]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY' \
    -- . ':(exclude)scripts/verify_reproduction.sh'; then
  echo "A credential-like value is present in tracked content." >&2
  exit 1
fi

if git ls-files | grep -Eiq \
    '(^|/)(\.env($|\.)|id_(rsa|dsa|ecdsa|ed25519)($|\.)|credentials($|\.)|.*\.pem$)'; then
  echo "A credential-like filename is tracked." >&2
  exit 1
fi

if git ls-files | grep -Eiq \
    '(^|/)(\.codegraph|__pycache__|codegraph)(/|$)|\.bak$|backup|before_'; then
  echo "A cache or backup artifact is tracked." >&2
  exit 1
fi

git diff --check
echo "Reproduction verification passed ($z1_lfs_count Git LFS objects)."
