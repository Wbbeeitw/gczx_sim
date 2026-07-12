# LIBERO Long Task 6：白杯放盘，布丁放盘右侧

## 任务卡

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `6` |
| 任务 | `put the white mug on the plate and put the chocolate pudding to the right of the plate` |
| BDDL | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl` |
| 目标 1 | `porcelain_mug_1` → `plate_1` |
| 目标 2 | `chocolate_pudding_1` → `living_room_table_plate_right_region` |
| 干扰物 | `red_coffee_mug_1` |

正式标签同步读取 LIBERO 原生 `check_ontop` 状态：

```text
On(porcelain_mug_1, plate_1)
AND
On(chocolate_pudding_1, living_room_table_plate_right_region)
```

## 阶段语义

- `phase 0`：首次 task6 有效操作之前。
- `phase 1`：白杯或巧克力布丁首次被夹爪受控搬运之后。
- `phase 2`：任一正确子目标已稳定完成，正在完成另一个目标。
- `phase 3`：两个原生目标均满足且环境真实成功。

`B1` 是任一目标物体的首次受控移动；`B2` 是白杯在盘上或布丁在右侧区域任一谓词的首次稳定成立，支持任意完成顺序；`B3` 是两个谓词同时成立，终帧以 `env_success` 对齐。阶段保持单调；失败 episode 不得产生 phase 3。

布丁在盘上或盘子左侧均为错误位置，只写入 audit，不能触发 B2/B3。红咖啡杯接触也只写入 audit，不能触发任何边界。

## 验收

采集 5 条 probe 后检查阶段单调、成功均有 B3、失败均无 B3、`b3_consistent_with_success=true`，并用 phase overlay GIF 复核两个完成顺序、错误布丁位置及红杯干扰。
