# LIBERO Long Task 7：字母汤和奶油奶酪盒放入篮子

| 字段 | 内容 |
| --- | --- |
| LIBERO task id | `7` |
| 任务 | `put both the alphabet soup and the cream cheese box in the basket` |
| 目标 | `alphabet_soup_1`、`cream_cheese_1` → `basket_1_contain_region` |
| 干扰物 | `tomato_sauce_1`、`ketchup_1` |

正式标签使用两个原生 `check_contain` 谓词。B1 为任一目标首次受控搬运；B2 为任一目标稳定入篮，或两个目标共同受控并接近篮子的最终转运；B3 为两个目标均入篮且环境成功。干扰物接触只进入 raw trace，不能触发阶段。
