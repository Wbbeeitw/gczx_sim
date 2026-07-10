# LIBERO Long Task 3：黑碗放入底部抽屉并关闭

## 状态

task3 的正式标签使用 rollout 同步记录的 LIBERO privileged state。当前不使用 VLM
生成正式标签；未来仅在历史数据缺少 trace 或发现视觉异常时，在本目录添加独立的
boundary-VLM 复核脚本。

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `3` |
| 任务 | `put the black bowl in the bottom drawer of the cabinet and close it` |
| BDDL | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl` |
| 目标物体 | `akita_black_bowl_1` |
| 目标抽屉 | `white_cabinet_1_bottom_region` |
| 抽屉 joint | `white_cabinet_1_bottom_level` |
| 干扰物 | `wine_bottle_1`、`wine_rack_1` |
| 初始抽屉状态 | 打开 |
| 正式标签方法 | rollout 同步读取 LIBERO 原生子目标谓词 |

BDDL 的完成条件是：

```text
Close(white_cabinet_1_bottom_region)
AND
In(akita_black_bowl_1, white_cabinet_1_bottom_region)
```

黑碗是否在抽屉内通过底部抽屉的原生 `check_contain` 判定；抽屉开关通过原生
`is_open` / `is_close` 判定。不要用画面距离、柜体接触或抽屉外观近似这些目标。

## 阶段语义

- `phase 0`：reset 稳定、接近黑碗或抽屉、首次 task3 有效操作之前。
- `phase 1`：首次受控搬运黑碗，或首次实际带动底部抽屉之后。
- `phase 2`：黑碗已稳定位于底部抽屉内部，正在完成最终关闭抽屉。
- `phase 3`：黑碗仍在抽屉内，抽屉关闭，LIBERO 环境真实成功。

边界定义：

- `B1`：黑碗被夹爪受控移动，或底部抽屉 joint qpos 首次出现实际变化。
- `B2`：`check_contain(black_bowl)` 首次稳定为真；抽屉提前关闭不触发 B2。
- `B3`：抽屉关闭且黑碗在抽屉内，最终以 `env_success` 为完成真值。

阶段单调：黑碗掉落、重新抓取、抽屉被重新打开或策略重试都不会让 phase 回退。
失败 episode 不产生 `phase 3`。wine bottle 和 wine rack 的接触只写入 audit，不能触发
B1/B2。

## 产物与验收

正式数据目录是 `/data/libero_long/task3/`，输出：

```text
meta/semantic_trace_task3.parquet
meta/phase_progress_semantic_trace_task3.parquet
meta/semantic_trace_task3_audit.csv
meta/semantic_trace_task3_metadata.json
```

probe 和正式批次都要确认：阶段单调；成功 episode 都有 B3；失败 episode 没有 B3；
audit 中 `b3_consistent_with_success=true`；抽屉提前关闭不会触发 B2；以及干扰物接触
不改变阶段边界。
