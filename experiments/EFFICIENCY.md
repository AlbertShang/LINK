# TraffiKrig PEMS03：单轮训练效率实验

在仓库根目录运行：

```bash
python experiments/benchmark_efficiency.py
```

脚本串行运行 STAGANN 注册的 10 个 baseline、KITS 和LINK，固定使用 TraffiKrig PEMS03 Top-K=3。默认分别运行 `metadata` 和 `no_metadata`，共 24 项。STAGANN 的转换数据若缺失或过期，会先从 `pure_dataset_topk/pems03` 生成。

如果需要指定不同的 Python 环境和 GPU：

```bash
python experiments/benchmark_efficiency.py \
  --stagann-python /root/miniconda3/envs/stagann/bin/python \
  --kits-python /root/miniconda3/envs/stagann/bin/python \
  --link-python /path/to/main-env/bin/python \
  --gpus 0 --device cuda:0
```

可以用 `--models GCN KITS LINK`、`--modes metadata` 选择子集。默认命令使用各方法已有的训练超参数；发生显存不足时，可以分别用 `--stagann-batch-size`、`--kits-batch-size`、`--center-batch-size` 调整，但该行会标为 `custom_or_unknown`，不应作为“默认配置”结果。`--skip-existing` 可跳过已有 JSON 结果，`--dry-run` 只打印命令。

结果写入 `outputs/efficiency_pems03_topk3/epoch_times.csv`，每项另有原始 JSON 和日志，并记录设备、GPU 型号及 PyTorch 版本。表中的 `seconds` 是**默认训练配置下单个 epoch 的耗时**：baseline 是首轮训练的实测值；LINK分别测 Trend 和 Residual 各一轮，再按默认计划的 50/50 阶段轮数加权平均。`trend_seconds`、`residual_seconds` 保留分阶段实测值，`total_seconds` 是实测两轮之和。`steps` 同样是每轮平均更新次数，`total_steps` 是实测更新次数总和。计时包括数据迭代、前向、反向、参数更新和该轮学习率更新；GPU 在计时边界同步，运行时统一关闭 `CUDA_LAUNCH_BLOCKING`。数据预处理、模型构建、验证、测试和保存不计时。Residual 轮在一轮 Trend 训练后开始，主要用于计时，不代表收敛后的模型精度。

各方法的原始 epoch 含有不同数量的训练样本和参数更新。结果表同时记录 `steps` 与 `seconds_per_step`；比较总耗时时应连同训练工作量一起报告。效率实验不运行验证和测试，也不生成精度结果。

结果表还记录 `parameter_count`、`parameter_mib`、`buffer_mib` 和 `model_state_mib`。其中 `model_state_mib` 是参数与模型缓冲区的张量大小之和，不是磁盘 checkpoint 大小，也不包含优化器状态。`peak_gpu_allocated_mib` 是这一训练轮次内 PyTorch 张量的最大显存占用；`peak_gpu_reserved_mib` 是缓存分配器管理的最大显存。计时开始前会重置峰值统计，但已常驻 GPU 的模型和图数据仍计入峰值。LINK另列 Trend 和 Residual 各自的 allocated 峰值，总峰值取两阶段的最大值。CPU 运行时 GPU 显存列为空；OOM 失败项没有完整的峰值结果。已有 JSON 若缺少这些内存字段，`--skip-existing` 也会重新运行该项。
