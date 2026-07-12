# LIBERO Long Task 8：两只 moka pot 放上炉子

目标为 `moka_pot_1` 和 `moka_pot_2` 同时位于 `flat_stove_1_cook_region`。炉子在 BDDL reset 时已开启，仅记录 audit，不作为阶段边界。B1 为任一 pot 首次受控搬运；B2 为第一只 pot 稳定在 cook region 或最终转运准备；B3 为两只 pot 均在 cook region 且环境成功。失败 episode 不得产生 phase 3。
