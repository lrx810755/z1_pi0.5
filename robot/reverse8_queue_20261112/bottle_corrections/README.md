# 当前瓶子长等待样本12条重录

由2026-11-11的长停顿筛查清单定位：当前透视修复版水平接近基础示范10条，加新纠偏中的transport和close_transition各1条。原始来源、SHA256、网格、种子参考及异常区间见retake_tasks.json。旧记录保留，不自动覆盖或移除训练来源。

沿用当前瓶子水平接近专家；第一列保持原规划/执行分支。10条基础示范保留从Home开始的完整接近、抓取、搬运、放盒及撤离。2条纠偏保留原类别，transport的人为扰动有显式事件与训练掩码。

新录制须满足原物理成功、同步、图像、释放和撤离验收；保存前重新分析frames并拒收连续近静止≥8秒的等待，已标注人为扰动不计作异常等待。优先恢复原来源布局；失败时可按原实现轮换同网格的成功参考布局，实际来源和seed写入新元数据。

本阶段在160条基础重录完成后顺序执行，整轮目标160+10+2=172条，不并发启动两个Isaac。不启动微调。

数据：`/home/lp531/smolvla_data/bottle_reverse8_corrections4_20261112`。日志：`/home/lp531/z1_queue_logs/bottle_reverse8_corrections4_20261112`。入口：`bash /home/lp531/SH/start_bottle_wait8_retake12_20261111.sh`。ROS域142、端口8854。master：`/home/lp531/robot/bottle_rerecord160_renderfix_20261111/run_queue.py`。
