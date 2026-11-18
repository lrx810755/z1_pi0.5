# =========================
# debug_z1_usd_final.py
# Isaac Sim 5.1+ 内网可用调试脚本
# =========================

from isaacsim import SimulationApp

# -------------------------
# 1. 先实例化 SimulationApp
# -------------------------
simulation_app = SimulationApp({"headless": False})

# -------------------------
# 2. 再导入其他模块
# -------------------------
import math
import numpy as np
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.types import ArticulationAction
import os
from z1_paths import asset_root

# =========================
# 3. 配置 USD 文件路径
# 注意：这里必须是已经导入 URDF 并保存的 USD 文件
# =========================
USD_PATH = str(asset_root() / "z1.usd")
STAGE_PATH = "/World/z1_debug"

if not os.path.exists(USD_PATH):
    raise FileNotFoundError(f"USD 文件不存在: {USD_PATH}")

# =========================
# 4. 创建世界
# =========================
world = World(stage_units_in_meters=1.0)

# -------------------------
# 5. 内网调试，不添加默认地面，避免 S3 下载
# world.scene.add_default_ground_plane()
# -------------------------

# =========================
# 6. 加载 USD
# =========================
stage_utils.add_reference_to_stage(
    usd_path=USD_PATH,
    prim_path=STAGE_PATH
)
world.reset()

# Make the standalone debug scene visible immediately.
create_prim(
    "/World/DebugLight",
    "DistantLight",
    attributes={"inputs:intensity": 3000, "inputs:angle": 0.8},
)
set_camera_view(
    eye=np.array([1.8, -2.2, 1.4]),
    target=np.array([0.0, 0.0, 0.35]),
    camera_prim_path="/OmniverseKit_Persp",
)
for _ in range(10):
    world.step(render=True)

stage = world.stage

# =========================
# 7. 查找 Articulation Root
# =========================
articulation_paths = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_paths.append(prim.GetPath().pathString)

if not articulation_paths:
    print("❌ 没有找到 ArticulationRoot，请确认 USD 是通过 URDF Importer 导入生成的")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()
    raise SystemExit

ROBOT_PRIM_PATH = articulation_paths[0]
print(f"✅ 找到 ArticulationRoot: {ROBOT_PRIM_PATH}")

# =========================
# 8. 创建 SingleArticulation
# =========================
z1 = SingleArticulation(
    prim_path=ROBOT_PRIM_PATH,
    name="z1"
)
world.scene.add(z1)
world.reset()
z1.initialize()
controller = z1.get_articulation_controller()

# =========================
# 9. 打印关节信息
# =========================
print("\n========== Z1 DOF Info ==========\n")
print("Number of DOF:", z1.num_dof)
print("DOF names:")
for i, name in enumerate(z1.dof_names):
    print(f"[{i}] {name}")
print("\n=================================\n")

if z1.num_dof == 0:
    print("❌ 找到了 Articulation Root，但没有可控 DOF。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()
    raise SystemExit

# =========================
# 10. 初始化关节归零
# =========================
z1.set_joint_positions(np.zeros(z1.num_dof))
for _ in range(60):
    world.step(render=True)

# =========================
# 11. 简单正弦运动调试
# =========================
print("开始关节调试运动，关闭窗口退出。")
step_count = 0

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        continue

    t = step_count * world.get_physics_dt()
    target_positions = np.zeros(z1.num_dof)

    # 控制前6个关节，夹爪保持默认
    if z1.num_dof >= 1: target_positions[0] = 0.4 * math.sin(t)
    if z1.num_dof >= 2: target_positions[1] = 0.3 * math.sin(t * 0.8)
    if z1.num_dof >= 3: target_positions[2] = 0.3 * math.sin(t * 0.6)
    if z1.num_dof >= 4: target_positions[3] = 0.4 * math.sin(t * 1.0)
    if z1.num_dof >= 5: target_positions[4] = 0.25 * math.sin(t * 1.2)
    if z1.num_dof >= 6: target_positions[5] = 0.25 * math.sin(t * 1.4)

    controller.apply_action(ArticulationAction(joint_positions=target_positions))

    if step_count % 120 == 0:
        print("Target:", np.round(target_positions, 3))
        print("Current:", np.round(z1.get_joint_positions(), 3))

    step_count += 1

simulation_app.close()
