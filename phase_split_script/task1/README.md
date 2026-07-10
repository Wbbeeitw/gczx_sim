# LIBERO Long Task 1：两个指定物体放入篮子

## 状态

已完成正式采集与特权状态阶段标注，当前是后续 task 的参考实现。现有脚本保持在
`phase_split_script/` 根目录，暂不迁移，避免影响已验证的服务器命令和 import。

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `1` |
| 任务 | `put both the cream cheese box and the butter in the basket` |
| 目标物体 | cream cheese box、butter |
| 目标容器 | basket |
| 正式标签方法 | rollout 同步记录 LIBERO simulator privileged state |
| 视觉路线 | `task1_boundary_vlm.py`，仅作历史数据 fallback、交叉复核和异常检查 |

## 阶段语义

- `phase 0`：reset 稳定、接近和首次有效操作之前。
- `phase 1`：首次明确抓住、抬起或受控拖动任一目标物体之后；包含掉落、重新抓取、
  单物体或双物体操作，不回退到 `phase 0`。
- `phase 2`：一个目标物稳定进入篮子，或两个目标物联合受控并到达篮子附近、进入最终
  转运准备。
- `phase 3`：LIBERO 环境真实任务成功。

边界分别为 `B1`（首次受控）、`B2`（部分完成或双物体最终转运）和 `B3`（完成）。
失败 episode 不产生 `B3`；成功结束帧的 `env_success` 是有效的特权完成证据。

## 现有实现与产物

- rollout 入口：`examples/recap/process/collect_libero_rollouts.py`
- 特权状态记录和状态机：`rlinf/revalue/semantic_trace.py`
- 旧规则标注器：`phase_split_script/task1.py`
- 边界 VLM 复核器：`phase_split_script/task1_boundary_vlm.py`
- 边界转 phase 工具：`phase_split_script/boundaries_to_phase_labels.py`
- 可视化工具：`phase_split_script/visualize_phases.py`

正式数据目录为 `/data/libero_long/task1/`，正式标签使用：

```text
meta/semantic_trace_task1.parquet
meta/phase_progress_semantic_trace_task1.parquet
meta/semantic_trace_task1_audit.csv
meta/semantic_trace_task1_metadata.json
```

`task1_boundary_vlm.py` 的输出必须使用独立 annotation name，不能覆盖上述正式特权状态
标签。

## 已知运行特性

LIBERO 在成功后会立即终止，因此 phase 3 往往只有很短的尾部。审计应验证成功 episode
具有 `env_success_terminal` 或 `state_terminal_success` 的 `B3` 来源，而不是要求固定长度
的 phase 3。
