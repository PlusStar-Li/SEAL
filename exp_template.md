下面是一份根据我们这次对话里反复澄清、踩坑和改设计总结出来的 **「实验需求提示词模板」**。你下次开新实验时，把对应段落填上即可，能少很多来回对齐。

---

## 实验需求提示词模板（SEAL · continual self-edit forgetting）

```markdown
## 1. 实验目标（一句话 + 要回答的问题）

- **实验名称**：（如 continual_self_edit_gen_forgetting）
- **核心问题**：（如：在 continual merge 下，self-edit **生成质量**是否在 held-out 任务上退化？）
- **与已有实验的关系**：
  - [ ] 与 `single` 模式对照（仅改 inner 规模 / split_newlines）
  - [ ] 与 `continual_self_edits`（inner TTT 遗忘矩阵）对照
  - [ ] 与 `CPT.sh` 超参对齐
  - [ ] 其他：___

## 2. 模式与「single / CPT 对称性」（必填）

请明确：**train 和 val 的逻辑是否要与另一模式平行**，差别只允许在：

| 维度 | single | CPT（本实验） |
|------|--------|---------------|
| 每 task 文章数 | ___（通常 1） | ___（如 50 / 100 / 200） |
| split_newlines | True / False | True / False |
| k_completions | ___ | ___ |
| inner epochs / bs / ga | ___ | ___ |

- **Merge 训练**：每步是对 **1 篇文章** 还是 **N 篇文章聚合 corpus** 做 fresh SE + inner SFT？
- **Val 评估**：共几个 task？每个 task 是几篇文章？每个 checkpoint 是否 **每个 val task 都重新生成 SE + inner + 评测**？
- **矩阵形状**：`(n_merge + 1) × n_val`，还是其他？请写清楚行列含义。

> 反例（请避免模糊表述）：「CPT 下 n_val 不生效」「eval 只要一个 overall 列」——若你需要多列，请写「n_val=8 表示 8 个独立 N 篇 corpus task」。

## 3. 数据与采样

- **数据集路径**：___
- **n_merge**（train task 数）：___
- **n_val**（val task 数）：___
- **每 task 文章数**（`inner_sft_articles`）：___
- **文章去重规则**：train / val 是否 article-level 互不重叠？同一 title 可否重复？
- **数据量检查**：需要 `(n_merge + n_val) × inner_sft_articles ≤ |dataset|` 篇不重复文章（请心算或注明已检查）

## 4. 模型与 LoRA

- **基座模型**：___（如 Qwen/Qwen2.5-7B 或 models/iter1）
- **Inner LoRA**（TTT_server）：r=___ α=___ dropout=___ target_modules=___（默认 qwen2 为 q_proj+v_proj）
- **Merge 方式**：inner adapter merge 进基座 / 仅评测不 merge
- **是否与 outer RL（train_SFT.sh）混用**：是 / 否

## 5. 运行规模与优先级

- **n_sequences**：___（若时间紧，写「先 1 seq smoke，再扩到 N」）
- **阶段计划**：
  - [ ] Smoke：建议 `n_merge=2, n_val=2, articles=20, seq=1`
  - [ ] Pilot：___
  - [ ] Full：___
- **可接受总时长**：___ 天；超时则优先砍：sequences / n_val / articles / epochs（请排序）

## 6. 产出物

- [ ] `summary_*.json`（正常结束自动生成）
- [ ] 热力图路径：___
- [ ] 若可能中途停止：需要从 `inner_seq{S}_row{R}_val{J}.log` 重建 summary
- [ ] 与 single / continual_self_edits 的 combined 对比图

## 7. 指标定义（避免歧义）

- **矩阵每个格子**表示什么？（如：该 checkpoint 上，对 val task j **fresh SE + inner TTT** 后的 **adapter overall acc**）
- **row 0「base」**是否仍做 inner SFT？（本仓库默认：**是**，base 指 0 次 merge 的生成器，不是零样本）
- **评测**：per-passage acc / corpus overall acc / 其他

## 8. 脚本与路径

- **入口脚本**：___（如 `general-knowledge/scripts/continual_self_edit_gen_forgetting.sh`）
- **INNER_MODE**：single / cpt
- **OUTPUT_DIR**：___（如 `.../cpt/run1`，避免覆盖已有 run）
- **INDEX / SEED / GPU**：___

## 9. 非功能需求

- [ ] 改代码 + 更新 README
- [ ] 只改 shell 配置
- [ ] 提供 log → summary 恢复脚本
- [ ] 运行时间预估

## 10. 参考（可选）

- 对齐论文 / 表格 / 图：___
- 类似历史 run：___（如 single/run0 已完成 8 seq）
```

---

## 精简版（日常快速提交）

若对话已经很长，可以只贴这一段：

```markdown
我要做【实验名】，目标是【科学问题】。

**对称性**：与【single/CPT】平行；train = n_merge 个 task（每 task 【N】篇），val = n_val 个 task（每 task 【N】篇）；每 checkpoint 对每个 val task 都 fresh SE + inner + eval。矩阵 (n_merge+1)×n_val。

**配置**：dataset=___，model=___，inner_sft_articles=___，k=___，epochs/bs/ga=___，split_newlines=___，n_merge=___，n_val=___，n_sequences=___。

**产出**：OUTPUT_DIR=___；要 summary + 热力图；【可能中途停，需从 log 恢复】。

**约束**：最多跑 ___ 天；数据共 ___ 篇；不要覆盖 run___。
```

---

## 我们这次对话里值得写进模板的「硬规则」

1. **写清「task」**：一个 task = 几篇文章、merge 一步用几个 task、val 几列各代表什么。  
2. **写清 CPT 下 `n_val` 是否生效**（你的全量设计里是 **8 个独立 corpus 列**）。  
3. **写清 row0 仍做 inner**（不是裸 base QA）。  
4. **写清是否会中途停** → 是否要 log 重建 summary。  
5. **写清数据上界**：`(n_merge+n_val)×articles ≤ |dataset|`。  
6. **写清 smoke / full 分阶段**，避免一上来 80 轮/seq × 多 seq。  
7. **OUTPUT_DIR / INDEX**，避免覆盖 smoke 的 `run0`。

---

## 填好后的示例（对应你最终定的 CPT forgetting）

```markdown
我要做 continual_self_edit_gen_forgetting 的 CPT 全量（seq0 已跑完，后续可能不跑多 seq）。

目标：在 CPT 配置（100 篇/task、k=5、split_newlines=0）下，测 continual merge 是否导致 held-out 上 fresh SE+inner 的 adapter acc 下降；与 single 的 (9×8) 矩阵结构一致。

Train：8 个 merge task，每 task 100 篇聚合 inner SFT 后 merge adapter。
Val：8 个 task，每 task 100 篇；每个 row（含 row0 base）对 8 个 val task 各 fresh SE + inner + overall acc。
矩阵 9×8；metric = adapter acc after fresh SE + TTT。

数据：squad_val.json；(8+8)×100=1600 篇。model=Qwen2.5-7B；inner LoRA r=32 α=64（TTT 默认 q/v）。
n_sequences=1（仅从 log 画图）；OUTPUT_DIR=cpt/run0；从 inner_seq0_row*_val*.log 重建 summary。
```

你可以把上面模板存成 `experiment_request.md` 或贴进 Cursor rule；需要我帮你改成英文版或压成一条 Cursor User Rule 时，切换到 Agent 模式即可。