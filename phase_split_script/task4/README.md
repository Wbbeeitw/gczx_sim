# LIBERO Long Task 4：两个杯子分别放到左右盘子

## 状态

task4 的正式标签使用 rollout 同步记录的 LIBERO privileged state。当前不使用 VLM
生成正式标签；未来仅在历史数据缺少 trace 或发现视觉异常时，在本目录添加独立的
boundary-VLM 复核脚本。

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `4` |
| 任务 | `put the white mug on the left plate and put the yellow and white mug on the right plate` |
| BDDL | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl` |
| 左盘目标 | `porcelain_mug_1` → `plate_1` |
| 右盘目标 | `white_yellow_mug_1` → `plate_2` |
| 干扰物 | `red_coffee_mug_1` |
| 正式标签方法 | rollout 同步读取 LIBERO 原生 `check_ontop` 谓词 |

BDDL 的完成条件是：

```text
On(porcelain_mug_1, plate_1)
AND
On(white_yellow_mug_1, plate_2)
```

两个目标都通过对应盘子的原生 `check_ontop` 判定。交叉放置（porcelain mug 放右盘、
white-yellow mug 放左盘）只进入 audit，不能触发 B2 或 B3。

## 阶段语义

- `phase 0`：reset 稳定、接近目标杯子、首次 task4 有效操作之前。
- `phase 1`：首次受控搬运任一目标杯子之后。
- `phase 2`：任一目标杯子已稳定放在自己的正确盘子上，正在完成另一个目标。
- `phase 3`：两个杯子都在各自正确盘子上，LIBERO 环境真实成功。

边界定义：

- `B1`：porcelain mug 或 white-yellow mug 首次被夹爪受控移动。
- `B2`：任一正确盘位谓词首次稳定为真，支持左右两种完成顺序。
- `B3`：两个正确盘位谓词都为真，最终以 `env_success` 为完成真值。

阶段单调：目标杯掉落、重新抓取、从盘子移开或策略重试都不会让 phase 回退。失败
episode 不产生 `phase 3`。red coffee mug 的接触只写入 audit，不能触发 B1/B2。

## 产物与验收

正式数据目录是 `/data/libero_long/task4/`，输出：

```text
meta/semantic_trace_task4.parquet
meta/phase_progress_semantic_trace_task4.parquet
meta/semantic_trace_task4_audit.csv
meta/semantic_trace_task4_metadata.json
```

probe 和正式批次都要确认：阶段单调；成功 episode 都有 B3；失败 episode 没有 B3；
audit 中 `b3_consistent_with_success=true`；交叉放错盘子不触发 B2；以及 red coffee
mug 接触不改变阶段边界。
