最终模型，30步预测；分别执行25步和20步后重新观测推理。
两组每组苹果/桃子/杯子/瓶子各25次，共200次。
execute_25和execute_20分别保存结果与三视角视频；失败默认记录failure_category/stop_reason/error_detail等。
comparison.json在每组完成后更新；job/status.json记录任务阶段，日志分组保存。
正常启动后不持续监控，不创建监控服务或自动重启。
最后关闭本次创建的场景。
