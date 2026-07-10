# LIBERO Long Task 2：打开炉灶并放置 moka pot

## 状态

task2 的正式标签使用 rollout 同步记录的 LIBERO privileged state。当前不使用 VLM
生成正式标签；未来仅在历史数据缺少 trace 或发现视觉异常时，在本目录添加独立的
boundary-VLM 复核脚本。

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `2` |
| 任务 | `turn on the stove and put the moka pot on it` |
| BDDL | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl` |
| 目标物体 | `moka_pot_1` |
| 炉灶 | `flat_stove_1` |
| 炉灶区域 | `flat_stove_1_cook_region` |
| 干扰物 | `chefmate_8_frypan_1` |
| 正式标签方法 | rollout 同步读取 LIBERO 原生子目标谓词 |

BDDL 的完成条件是：

```text
Turnon(flat_stove_1)
AND
On(moka_pot_1, flat_stove_1_cook_region)
```

`Turnon` 从 `flat_stove_1_button` joint 的真实 qpos 导出；moka pot 的放置状态通过
`flat_stove_1_cook_region` 的原生 `check_ontop` 判定。不要用图像距离、物体接触或
炉灶外观近似这两个目标。

## 阶段语义

- `phase 0`：reset 稳定、接近和首次 task2 有效操作之前。
- `phase 1`：首次受控搬运 moka pot，或首次实际带动炉灶按钮之后。
- `phase 2`：已完成一个真实子目标，正在完成另一个子目标。
- `phase 3`：炉灶开启且 moka pot 位于 cook region，LIBERO 环境真实成功。

边界定义：

- `B1`：moka pot 被夹爪受控移动，或炉灶按钮 qpos 首次出现实际变化。
- `B2`：`stove_turn_on` 或 `moka_pot_on_cook_region` 首次稳定为真；支持“先开炉灶”
  和“先放 moka pot”两种有效顺序。
- `B3`：两个子目标同时满足，最终以 `env_success` 为完成真值。

阶段单调：moka pot 掉落、重新抓取、炉灶暂时被关或策略重试都不会让 phase 回退。
失败 episode 不产生 `phase 3`。frypan 的接触只写入 audit，不能触发 B1/B2。

## 产物与验收

正式数据目录是 `/data/libero_long/task2/`，输出：

```text
meta/semantic_trace_task2.parquet
meta/phase_progress_semantic_trace_task2.parquet
meta/semantic_trace_task2_audit.csv
meta/semantic_trace_task2_metadata.json
```

probe 和正式批次都要确认：阶段单调；成功 episode 都有 B3；失败 episode 没有 B3；
audit 中 `b3_consistent_with_success=true`；以及 frypan 接触不改变阶段边界。
