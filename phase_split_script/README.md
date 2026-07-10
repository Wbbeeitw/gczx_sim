# LIBERO Long 阶段标注工作区

这个目录用于管理 LIBERO Long `task0`--`task9` 的阶段标注工具与任务说明。
当前采用渐进式管理：已验证的脚本保留在根目录，新的任务专属脚本从各自的
`taskN/` 目录开始维护。不要为了整理目录而移动已经在服务器命令中使用的脚本。

## 当前布局

- 根目录的 `task1.py`、`task1_boundary_vlm.py`、`visualize_phases.py`、
  `boundaries_to_phase_labels.py` 和 `qwen_client.py` 是现有可执行工具，保持兼容。
- `task1/README.md` 记录 task1 已验证的采集、特权状态标注和视觉复核流程。
- `task2/`--`task9/` 是后续任务的专属工作区。开始一个任务时，将该任务的说明、
  专属标注器、审计器和必要的 VLM fallback 放入对应目录。
- `output_demo/` 仅放服务器生成的 GIF/视频，不能提交数据或可视化产物到 Git。

## 每个任务的固定流程

1. 在 `/data/libero_long/taskN/` 用 rollout collector 采集一个小型 probe。
2. 从 BDDL、probe 的 simulator state 和 GIF 确认对象别名、容器/目标和成功终止行为。
3. 填写 `taskN/README.md` 中的任务卡，并实现该任务的特权状态 trace 标注。
4. 为未带 trace 的历史数据或异常样本保留 boundary-VLM 复核路线；不使用整段视频
   一次性生成 phase 序列作为正式标签。
5. 在正式采集阶段同步输出 raw trace、phase parquet、audit CSV 和 metadata JSON。
6. 每批数据抽取成功与失败 episode 生成 GIF；确认阶段单调、成功有完成态、失败没有
   伪完成态后再进入 ReValue。

## 数据与命名约定

每个任务的数据根目录为 `/data/libero_long/taskN/`。机器可读的派生产物放
`meta/`，建议使用下面的名称：

```text
semantic_trace_taskN.parquet
phase_progress_semantic_trace_taskN.parquet
semantic_trace_taskN_audit.csv
semantic_trace_taskN_metadata.json
```

VLM 边界复核结果使用独立名称，例如
`phase_progress_semantic_boundary_v1.parquet`，不能覆盖特权状态正式标签。GIF 输出到：

```text
/workspace/RLinf/phase_split_script/output_demo/<annotation_name>/
```

## 开始新任务前必须确认的内容

- 任务自然语言、BDDL、目标物体和目标容器/区域。
- B1（首次有效受控）、B2（进入最终转运/部分完成）和 B3（环境成功）的任务语义。
- 可读取的 simulator privileged state：object pose、gripper、contact、near-target、
  环境 success 等。
- LIBERO 在 success 后是否立即终止；这决定 phase 3 的尾部记录策略。
- probe 上至少一条成功和一条失败轨迹的 audit 与 GIF 是否符合预期。

不要预先创建空的 `annotate.py`、`audit.py` 或 VLM 脚本。不同任务的对象关系和成功
条件不同，等 task card 与 probe 确认后再添加实际实现。
