# 源快照状态

整理日期：2026-08-31

## LeRobot

- 来源：`https://github.com/huggingface/lerobot.git`
- 提交：`0878c688`
- 原仓库处于 detached HEAD；
- 原仓库未跟踪的嵌套 `LIBERO/` 未复制到本仓库。

## z1_ros2

- 来源：`https://github.com/idra-lab/z1_ros2.git`
- 上游基线：`d2462f735bd20cfd45c22692f9ddfbc3c7bb1cd5`
- 当前功能包含大量未提交修改和新增文件；
- 本仓库复制的是工作树实际内容，而不是只依赖上游提交；
- `build/`、`install/`、`log/`、缓存和原 `.git/` 未复制。

## Isaac 资产

- 原位置：`/home/lp531/isaac_assets/z1`；
- 原目录不是有效 Git 仓库；
- 已复制场景、脚本、配置、生成资产、扫描模型及工作记录；
- CodeGraph、Codex 元数据和 Python 缓存未复制。

## 标定

实际使用的手眼标定同时保存在：

```text
calibration/z1_handeye_charuco_gripper_to_camera.yaml
ros2_ws/z1_handeye_charuco_gripper_to_camera.yaml
```

第二份保留原运行布局，第一份便于独立查找。

