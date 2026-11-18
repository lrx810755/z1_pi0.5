# 数据说明

现有数据根目录：

```text
/home/lp531/smolvla_data
```

当前检查到：

- 19 个带成功摘要的 Episode：17 个苹果、2 个桃子；
- 其中包含隔离数据，成功状态不等于已经完成人工质量审核；
- 13 个未完成的 `.recording_*` 目录；
- 饮料瓶尚无通过完整端到端物理门禁的有效 Episode；
- 一个可由 LeRobot 0.3.4 加载的实验转换集，共 1 个 Episode、405 帧、2 路相机。

因此代码仓库不直接包含数据文件。迁移时应单独归档整个 `smolvla_data`，正式训练时再从人工审核通过的 Episode 构建只读数据版本。

建议后续结构：

```text
smolvla_data/
├── raw/             # 原始录制，只读保留
├── quarantine/      # 无效、可疑或未完成回合
├── reviewed/        # 人工审核通过的数据
├── lerobot/         # 转换后的版本化数据集
└── manifests/       # Episode 清单、审核结果和哈希
```

