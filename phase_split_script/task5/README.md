# LIBERO Long Task 5：黑书放入笔筒后格

## 状态

task5 的正式标签使用 rollout 同步记录的 LIBERO privileged state。当前不使用 VLM
生成正式标签；未来仅在历史数据缺少 trace 或发现视觉异常时，在本目录添加独立的
boundary-VLM 复核脚本。

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `5` |
| 任务 | `pick up the book and place it in the back compartment of the caddy` |
| BDDL | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl` |
| 目标物体 | `black_book_1` |
| 目标容器 | `desk_caddy_1_back_contain_region` |
| 干扰物 | `white_yellow_mug_1` |
| 正式标签方法 | rollout 同步读取 LIBERO 原生 containment 谓词 |

BDDL 的完成条件是：

```text
In(black_book_1, desk_caddy_1_back_contain_region)
```

最终完成只通过后格原生 `check_contain(black_book)` 与 `env_success` 判定。B2 另外记录
书本到后格 site 的距离及书本-笔筒 contact，解决 containment 与终止同帧时 phase 2 为空的
问题。

## 阶段语义

- `phase 0`：reset 稳定、接近黑书、首次 task5 有效操作之前。
- `phase 1`：黑书首次被夹爪受控搬运之后。
- `phase 2`：黑书进入笔筒后格的最终插入准备区，正在完成最后插入动作。
- `phase 3`：黑书位于后格，LIBERO 环境真实成功。

边界定义：

- `B1`：黑书被夹爪闭合、接触且受控移动。
- `B2`：黑书距离后格 site 小于等于候选阈值 `0.10m`，且仍受控或已与笔筒接触。
- `B3`：原生 `check_contain(black_book)` 为真，最终以 `env_success` 为完成真值。

`0.10m` 只用于 B2，必须通过 5 条 probe 的 raw distance、audit 与视频校准；它不影响
B3 的真实成功判定。阶段单调：书本掉落、重新抓取或重新对准都不会让 phase 回退。白黄杯
接触只写入 audit，不能触发 B1/B2。

## 产物与验收

正式数据目录是 `/data/libero_long/task5/`，输出：

```text
meta/semantic_trace_task5.parquet
meta/phase_progress_semantic_trace_task5.parquet
meta/semantic_trace_task5_audit.csv
meta/semantic_trace_task5_metadata.json
```

probe 必须确认：阶段单调；成功 episode 都有 B3；失败 episode 没有 B3；audit 中
`b3_consistent_with_success=true`；B2 出现于书本接近并插入后格的最后过程，而不是刚拿起书；
白黄杯接触不改变阶段边界。
