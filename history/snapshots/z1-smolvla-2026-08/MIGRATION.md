# 迁移说明

## 生成代码与资产包

```bash
cd /path/to/z1-smolvla
./scripts/package_repository.sh /path/to/smolvla_migration
```

脚本只生成一个核心仓库压缩包 `tar.zst`。压缩包内自带顶层 `z1-smolvla/` 目录，不包含 Git 元数据、训练输出、数据载荷、ROS `build/install/log`、历史工作日志、旧版单相机场景和备份资产。

## 单独归档现有数据

```bash
mkdir -p /path/to/smolvla_migration

tar -I 'zstd -T0 -6' \
  -cf /path/to/smolvla_migration/smolvla-data-full.tar.zst \
  -C /path/to/data-parent \
  smolvla_data
```

完整数据归档包含失败试验和未完成录制，只用于迁移保全，不代表正式训练集。

## 目标 PC 布局

核心仓库可以解压到任意位置，不需要 GitHub，也不需要初始化 Git：

```bash
tar -I zstd -xf z1-smolvla-core-YYYY-MM-DD.tar.zst
cd z1-smolvla
source scripts/setup_env.sh
./scripts/verify_reproduction.sh
```

如 Isaac Sim 或数据不在默认位置，设置 `ISAAC_SIM_ROOT` 和 `SMOLVLA_DATA_ROOT`。详细步骤见 `REPRODUCE.md`。

Isaac Sim、ROS 2、MoveIt、NVIDIA 驱动、Conda 环境和基础 SmolVLA checkpoint 需要在目标 PC 重新安装。
