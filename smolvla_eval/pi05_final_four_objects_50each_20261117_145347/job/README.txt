本次评估：最终100000次更新模型，苹果/桃子/杯子/瓶子各50次，共200次。
任务进程正常启动后助手不持续监控；不创建监控服务或自动重启。
status.json保存任务阶段与完成/失败状态。
evaluation.log保存评估日志，scene.log保存场景日志，final_model_verification.log保存GPU核验日志。
上级目录pi05_sim_eval_时间/report.json逐条更新成功率与失败分类，episodes中保存逐条result.json及trace、输入同步、夹爪预测和三视角视频。
失败默认写failure_category/stop_reason/error_detail/cleanup_error及物理证据；不把失败样本删除或重录。
任务结束后只清理本次创建的评估与场景子进程。
