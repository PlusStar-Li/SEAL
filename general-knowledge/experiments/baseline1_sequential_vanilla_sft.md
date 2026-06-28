# Baseline 1：Sequential Vanilla SFT（下界对照）

> 实验规格文档 · 用于下一步代码生成  
> 所属项目：Adaptive O-LoRA with Semantic Self-Reflection  
> 模板来源：`exp_template.md`

---

## 1. 实验目标（一句话 + 要回答的问题）

- **实验名称**：`baseline1_sequential_vanilla_sft`
- **核心问题**：在 **single-passage continual merge** 下，使用 **RL 对齐后的 SEAL 模型（`models/iter2`，含 $W_{SE}$）** 作为起点，vanilla inner LoRA SFT + merge **是否仍会导致 held-out 任务上 fresh SE + inner TTT 的 adapter QA 准确率下降**（self-edit 生成能力 / 下游可用性退化）？
- **Baseline 角色**：公平对照矩阵中的 **下界（Lower Bound）** —— 无正交约束、无任务相似度感知、无参数复用。
- **与已有实验的关系**：
  - [x] 与 `continual_self_edit_gen_forgetting` **single 模式** 完全对齐（仅改初始模型与输出路径）
  - [x] 其他：作为 Baseline 2/3/Proposed 的 **共同起点**（统一 `models/iter2`）

---

## 2. 模式与「single / CPT 对称性」

本 baseline **仅 single passage**；与 Baseline 2/3 保持 **train/val 逻辑完全平行**。

| 维度 | 本实验（single） | CPT（不做） |
|------|------------------|-------------|
| 每 task 文章数 | **1** | — |
| split_newlines | **True** | — |
| k_completions | **1** | — |
| inner epochs / bs / ga | **10 / 1 / 1** | — |

- **Merge 训练**：共 `n_merge` 步；每步对 **1 篇 train 文章** 做 fresh SE → inner TTT（vanilla LoRA）→ **merge adapter 进当前基座**。
- **Val 评估**：共 `n_val` 个 **held-out 单篇 task**（与 merge train 文章 **article-level 互不重叠**）；**每个 merge checkpoint（含 row 0）** 对每个 val task **重新** fresh SE + inner TTT + 评测。
- **矩阵形状**：`(n_merge + 1) × n_val`
  - **行（row）**：row 0 = 0 次 merge 的生成器 checkpoint；row r = 完成 r 次 merge 后的 checkpoint（r = 1..n_merge）。
  - **列（col）**：val task j（j = 0..n_val-1），每列对应一篇 held-out 文章及其 QA。
- **row 0 含义**：**不是** zero-shot QA；row 0 仍对 val 做 **fresh SE + inner TTT**，只是 merge 次数为 0。

---

## 3. 数据与采样

- **数据集路径**：`general-knowledge/data/squad_val.json`
- **n_merge**（train / merge stream task 数）：**3**（Full）；Smoke：**2**
- **n_val**（held-out eval task 数）：**3**（Full）；Smoke：**2**
- **每 task 文章数**（`inner_sft_articles`）：**1**
- **文章去重规则**：
  - 同一 sequence 内，train 与 val **article-level 互不重叠**（`(title, context)` 为唯一键，title 可重复）。
  - 采样逻辑复用 `split_train_val_disjoint()`。
- **数据量检查**：`(n_merge + n_val) × 1 ≤ |dataset|` → Full 需 **16** 篇；`squad_val.json` 足够。
- **跨 baseline 公平性（硬约束）**：
  - 与 Baseline 2/3 使用 **相同 `--seed`、相同 `n_merge/n_val/n_sequences`**，保证 **同一 seq 索引对应同一 train/val split**（复用 `splits/seq*/{train,val}.json` 或同 seed 重采样）。

---

## 4. 模型与 LoRA

- **基座模型（统一起点）**：`models/iter2`
  - 含义：经 ReST-EM outer RL（`train_SFT.sh`）训练并 merge 后的 checkpoint；**Self-Edit 生成能力已由 $W_{SE}$ 对齐**，作为所有 baseline 的共同初始生成器。
  - **禁止**使用裸 `Qwen/Qwen2.5-7B` 作为本组对照的起点（避免与历史 `single/run0` 混淆）。
- **Inner LoRA（TTT_server，vanilla）**：
  - r=**32**, α=**64**, dropout=**0**
  - target_modules：**PEFT 默认**（Qwen2 → `q_proj`, `v_proj`）
  - **无**正交损失；**无**任务 bank；**无** $\lambda_t$
- **Merge 方式**：每 merge 步将 inner adapter **`merge_and_unload` 进当前基座**（与 `continual_self_edit_gen_forgetting` / `continual_self_edits` 一致）。
- **是否与 outer RL 混用**：**否**（本实验仅 **消费** iter2，不再训练 outer loop）。
- **$W_{SE}$ 在本 baseline 中的处理**：已 merge 进 `models/iter2` 权重；inner loop **不单独挂载/冻结** meta LoRA，行为与现有 SEAL continual 实验一致。

---

## 5. 运行规模与优先级

- **n_sequences**：**5**（Full，与当前脚本默认一致）；Smoke：**1**
- **阶段计划**：
  - [x] **Smoke**：`n_merge=2, n_val=2, n_sequences=1, model=models/iter2` — 验证 iter2 路径、split、summary、热力图
  - [ ] **Pilot**：`n_merge=4, n_val=4, n_sequences=2`
  - [x] **Full**：`n_merge=8, n_val=8, n_sequences=5`
- **可接受总时长**：约 **2–3 天/seq**（与历史 single/run0 同量级）；超时优先砍：**n_sequences → n_val → n_merge**
- **与历史 run 关系**：
  - 已有 `continual_self_edit_gen_forgetting/single/run0` 使用 **Qwen base**；**不要覆盖**。
  - 本 baseline 使用 **新 OUTPUT_DIR**（见 §8）。

---

## 6. 产出物

- [x] `summary_<timestamp>.json`（`continual_self_edit_gen_forgetting` 正常结束自动生成）
- [x] `splits/seq*/{train,val,index}.json`（split manifest）
- [x] 热力图：`forgetting_heatmap_se_gen.png`
- [ ] combined 对比图（可选）：若同期跑 iter2 版 `continual_self_edits`，可 `--inner_results_dir` 并排
- [ ] log → summary 重建（若中途停止）：从 driver log / per-val 打印行重建（与 cpt/run0 做法一致）
- [x] **额外分析字段（建议写入 summary 或后处理）**：
  - 每 row 的 **mean val acc**（跨列平均）
  - 对角/非对角遗忘趋势（可选，主矩阵已是全 val 列）

---

## 7. 指标定义（避免歧义）

| 指标 | 定义 |
|------|------|
| **主矩阵格子 $(r,j)$** | 在 checkpoint $r$（$r$ 次 merge）上，对 val task $j$：**fresh SE（k=1）→ vanilla inner TTT → adapter accuracy**（GPT-4.1 判分，与现有 `TTT_server` 一致） |
| **row 0** | **仍做 inner SFT**；表示 iter2 生成器在 **0 次 continual merge** 下的 SE+TTT 表现 |
| **下游 QA** | 上表 adapter accuracy（per-passage，该 val 文章全部 questions 平均） |
| **Self-Edit 生成质量** | **本 baseline 代码路径暂不自动算 Rouge-L/格式率**；若需与 Proposed 对齐，后处理脚本对保存的 completion 计算（见 §9 扩展） |
| **参数空间复用率** | **N/A**（vanilla 每步新开 LoRA 并 merge，无复用统计） |

---

## 8. 脚本与路径

- **入口脚本（复用，仅改配置）**：`general-knowledge/scripts/continual_self_edit_gen_forgetting.sh`
- **INNER_MODE**：`single`
- **关键 shell 变量**：

```bash
INNER_MODE="single"
MODEL_NAME="models/iter2"
DATASET="general-knowledge/data/squad_val.json"
OUTPUT_DIR="general-knowledge/results/baselines/baseline1_vanilla_sft/run${INDEX}"
N_SEQUENCES=3
N_MERGE=8
N_VAL=8
# single preset 自动：INNER_SFT_ARTICLES=1, K_COMP=1, EPOCHS=10, BS=1, GA=1, SPLIT_NEWLINES=1
```

- **Python 模块（无新代码）**：`general-knowledge.src.continual.continual_self_edit_gen_forgetting`
- **INDEX / SEED / GPU**：`INDEX=0`, `SEED=42+INDEX`, GPU `0,1`（vLLM + inner）
- **绘图**：`plot_self_edit_gen_forgetting.py --results_dir ${OUTPUT_DIR}`

---

## 9. 非功能需求（代码生成阶段）

- [ ] **本 baseline：仅改 shell + 文档**，不改 Python（前提：`models/iter2` 路径可被 vLLM / HF 正常加载）
- [ ] 更新 `general-knowledge/README.md`：增加 Baseline 1 小节与 OUTPUT_DIR 说明
- [ ] 确认 `models/iter2` 存在且为 merge 后完整权重目录
- [ ] 直接启动 Full，不需要Smoke
- [ ] **禁止覆盖** `continual_self_edit_gen_forgetting/single/run0`

### 9.1 建议后续扩展（非本 baseline 阻塞项）

- 保存每步 SE completion 文本 → 离线算 **Rouge-L（vs passage）**、**Implications 格式率**
- 与 Baseline 2/3 共用 **同一 split manifest** 的 symlink 或 `--splits_dir` 参数（需在 Baseline 2/3 开发时一并设计）

---

## 10. 参考

- **对齐论文/图**：SEAL Sec.5 continual self-edits；本实验为 **SE-gen forgetting** 变体（held-out val 全矩阵）
- **类似历史 run**：`continual_self_edit_gen_forgetting/single/run0`（**Qwen base**，非 iter2 — 仅作流程参考，**不可直接对比数值**）
- **方法论文档**：项目根目录 Adaptive O-LoRA 大纲 §4 Baseline 1

---

## 11. 代码生成清单（下一步）

| 项 | 动作 |
|----|------|
| Shell | 复制 `continual_self_edit_gen_forgetting.sh` → `baseline1_vanilla_sft.sh`，固定 `MODEL_NAME=models/iter2` 与新 OUTPUT_DIR |
| Python | **无** |
| 数据 | 确认 iter2；dataset 不变 |
---


