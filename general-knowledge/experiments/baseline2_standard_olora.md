# Baseline 2：Standard O-LoRA（恒定强正交，λ ≡ 1.0）

> 实验规格文档 · 用于下一步代码生成  
> 所属项目：Adaptive O-LoRA with Semantic Self-Reflection  
> 模板来源：`exp_template.md`

---

## 1. 实验目标（一句话 + 要回答的问题）

- **实验名称**：`baseline2_standard_olora`
- **核心问题**：在 SEAL continual single-passage 流程中，对 **每个新任务** 的 inner LoRA 施加 **恒定满强度正交约束（$\lambda_t \equiv 1.0$）**、并维护历史正交子空间 $\mathcal{U}_{hist}$，能否 **缓解** Baseline 1 的 SE 能力退化？同时是否会暴露 **标准 O-LoRA 的局限**（参数空间随任务增加迅速枯竭、相似任务无法充分共享子空间）？
- **Baseline 角色**：公平对照矩阵中的 **Standard O-LoRA 基线** —— 有正交、**无**相似度感知、**无**任务 LoRA 复用（因 $\lambda \equiv 1 \ge \tau$，恒开新 $W_{task_t}$）。
- **与已有实验的关系**：
  - [x] 与 Baseline 1 **同一 eval 矩阵协议**（`(n_merge+1)×n_val`，fresh SE + inner + adapter acc）
  - [x] 与 Baseline 1 **同一初始模型** `models/iter2`
  - [ ] 与 `continual_self_edit_gen_forgetting`：**需新 inner 训练逻辑**（O-LoRA loss），driver 可 fork 扩展
  - [x] 其他：为 Proposed（自适应 $\lambda_t$ + 复用）提供 **「强正交但无自适应」** 对照

---

## 2. 模式与「single / CPT 对称性」

与 Baseline 1 / 3 **完全平行**（仅 inner 训练 loss 与 task 管理不同）。

| 维度 | 本实验 | Baseline 1 | Baseline 3 |
|------|--------|------------|------------|
| 每 task 文章数 | 1 | 1 | 1 |
| split_newlines | True | True | True |
| k_completions | 1 | 1 | 1 |
| inner epochs / bs / ga | 10 / 1 / 1 | 同左 | 同左 |
| $\lambda_t$ | **固定 1.0** | 无 | embedding 自适应 |
| 任务 LoRA 复用 | **否**（恒新开） | N/A | 是（$\lambda<\tau$） |
| 正交子空间 $\mathcal{U}_{hist}$ | **是，累积** | 否 | 是，累积 |

- **Merge 训练**：每步 1 篇 train 文章 → fresh SE → **O-LoRA inner SFT**（$\lambda=1$）→ merge。
- **Val 评估**：与 Baseline 1 相同；**每个 checkpoint × 每个 val task** fresh SE + **同一 O-LoRA inner 配置** + eval。
- **矩阵形状**：`(n_merge + 1) × n_val`（行列语义同 Baseline 1）。

---

## 3. 数据与采样

- **数据集路径**：`general-knowledge/data/squad_val.json`
- **n_merge**：**8**（Full）；Smoke：**2**
- **n_val**：**8**（Full）；Smoke：**2**
- **inner_sft_articles**：**1**
- **去重与 split**：同 Baseline 1；**必须与 Baseline 1/3 同 seed 同 split**（公平对比硬约束）。
- **数据量检查**：16 篇（Full）。

---

## 4. 模型与 LoRA

### 4.1 基座与 $W_{SE}$（Task 0 投影）

- **Continual 起点模型**：`models/iter2`（与 Baseline 1 相同）。
- **$\mathcal{U}_{hist}$ 初始化（Task 0 投影，与方法论一致）**：
  - 从 iter2 所对应的 **outer RL LoRA（$W_{SE}$）** 提取 **A 矩阵**（按 O-LoRA 论文：对每层 target module 的 LoRA $A$）。
  - **实现约定（需在代码中明确其一并写死）**：
    - **方案 A（推荐）**：若 `models/iter2` 为 **已 merge 全量权重**、不保留独立 adapter 目录，则 **额外加载** outer 训练最后一轮 adapter checkpoint（如 `models/iter2_lora_adapter/` 或 RL 产物路径），仅读取 **A** 写入 $\mathcal{U}_{hist}\leftarrow\{U_{SE}\}$。
    - **方案 B（退而求其次）**：以 iter2 上 **第一次 inner TTT** 产生的 $A$ 作为 $U_{SE}$ 代理，并在文档中标注与方法论偏差。
  - **代码生成前待确认**：仓库中 **$W_{SE}$ adapter 磁盘路径**（若无，需补存或采用方案 B 并做 sensitivity）。

### 4.2 Inner task LoRA（$W_{task_t}$）

- r=**32**, α=**64**, dropout=**0**；target_modules：与 vanilla inner 一致（**`q_proj`, `v_proj`**），保证与 Baseline 1 **同容量**。
- **每个 merge 步**：**新初始化** $W_{task_t}$（$\lambda\equiv1\ge\tau$，不复用历史 task LoRA）。

### 4.3 正交损失（Standard O-LoRA，$\lambda_t=1$）

对当前 task 的 LoRA **A 矩阵** $A_t$ 与历史基 $\mathcal{U}_{hist}=\{U_0,\ldots,U_{t-1}\}$：

$$
\mathcal{L} = \mathcal{L}_{SFT}(P_t) + \gamma \cdot \mathcal{L}_{ortho}(A_t, \mathcal{U}_{hist})
$$

- **$\mathcal{L}_{SFT}$**：与现有 `TTT_server` 相同（causal LM on `build_train_sequences` 输出；mask 规则不变）。
- **$\mathcal{L}_{ortho}$**（实现需与 O-LoRA 原文一致，建议默认）：
  - 对每个 LoRA 层、每个历史 $U\in\mathcal{U}_{hist}$：**Frobenius 惩罚** $\|A_t U^\top\|_F^2$（或论文等价形式 $\|A_t U\|_F^2$，以选定 O-LoRA 实现为准）。
  - 对所有层、所有历史 $U$ **求和**。
- **$\gamma$（全局正交系数）**：**超参，默认 `1.0`**；Smoke 可扫 `{0.1, 1.0}`；Full 固定单一值并在 summary 记录。
- **$\lambda_t$**：**恒为 `1.0`**（CLI：`--lambda_fixed 1.0` 或 `--olora_mode standard`）；**不启用**相似度分支。

### 4.4 $\mathcal{U}_{hist}$ 更新规则

- **Task 0**：$U_{SE}$（见 §4.1）加入 $\mathcal{U}_{hist}$。
- **每完成一个 merge 步 $t$**（train 文章 $t$）：将本次 $W_{task_t}$ 的 **A 矩阵** 追加到 $\mathcal{U}_{hist}$（**不 merge A 进基座后再丢 B**；merge 仍按 SEAL 对 **完整 adapter** 做 `merge_and_unload` 进 $\Theta_{Base}$，与 Baseline 1 一致）。
- **Val eval 时的 $\mathcal{U}_{hist}$ 快照**：eval 使用 **截至当前 checkpoint 已完成 merge 步** 的 $\mathcal{U}_{hist}$（与 train 同步）；val 本身 **不向** $\mathcal{U}_{hist}$ 追加（除非设计为 eval 也训练 —— **否**）。

### 4.5 Merge

- 与 Baseline 1 相同：inner 训完 → adapter **merge 进当前生成器权重** → 下一步 vLLM 加载 merged 模型。

---

## 5. 运行规模与优先级

- **n_sequences**：**5**（Full）；Smoke：**1**
- **阶段计划**：
  - [x] **Smoke**：`n_merge=2, n_val=2, seq=1` + 验证 ortho loss 非零、$\mathcal{U}_{hist}$ 长度随步递增
  - [ ] **Pilot**：`n_merge=4, n_val=4, seq=2`
  - [x] **Full**：`n_merge=8, n_val=8, seq=5`
- **可接受总时长**：略高于 Baseline 1（正交项计算）；超时砍：**n_sequences → n_merge**
- **超参敏感性（可选 appendix）**：$\gamma \in \{0.1, 1.0, 10.0\}$ 仅 1 seq

---

## 6. 产出物

- [x] `summary_*.json`（含 `olora_mode=standard`, `gamma`, `lambda_fixed=1.0`, `U_hist_size_per_row`）
- [x] `splits/seq*/`（与 Baseline 1 同 seed 可共享）
- [x] 热力图：`forgetting_heatmap_se_gen.png`
- [x] **O-LoRA 专用日志**（建议）：
  - 每 merge 步：$\mathcal{L}_{SFT}$, $\mathcal{L}_{ortho}$, $\|\cdot\|_F$ 分项
  - $\mathcal{U}_{hist}$ 基数（Task 0 + 已完成 merge 步数）
- [ ] 参数复用率：**恒 0**（记录即可）

---

## 7. 指标定义

| 指标 | 定义 |
|------|------|
| **主矩阵 $(r,j)$** | 同 Baseline 1，但 inner 为 **Standard O-LoRA**（$\lambda=1$） |
| **row 0** | iter2 生成器 + **O-LoRA inner**（$\mathcal{U}_{hist}=\{U_{SE}\}$） |
| **下游 QA** | adapter accuracy（GPT-4.1） |
| **Self-Edit 质量** | 后处理 Rouge-L / 格式率（同 Baseline 1） |
| **参数空间复用率** | **0%**（本 baseline 设计） |
| **正交约束强度** | 报告 $\mathcal{L}_{ortho}$ 均值曲线 |

---

## 8. 脚本与路径

- **入口脚本（新建）**：`general-knowledge/scripts/baseline2_standard_olora.sh`
- **Python driver（新建，建议 fork）**：`general-knowledge/src/continual/baseline2_standard_olora.py`  
  - 或扩展 `continual_self_edit_gen_forgetting.py` + `--inner_mode o_lora_standard`（二选一，代码生成时定）
- **Inner server（新建或扩展）**：`general-knowledge/src/inner/TTT_server_olora.py`  
  - ZMQ 请求 **新增字段**：`ortho_enabled`, `lambda_t`, `gamma`, `U_hist`（或 `U_hist_path`）, `accumulate_A` 等
- **INNER_MODE 语义**：single + O-LoRA
- **OUTPUT_DIR**：`general-knowledge/results/baselines/baseline2_standard_olora/run${INDEX}`

### 8.1 建议 CLI 参数

```
--model models/iter2
--dataset general-knowledge/data/squad_val.json
--n_merge 8 --n_val 8 --n_sequences 5 --seed 42
--inner_sft_articles 1 --k_completions 1 --split_newlines
--finetune_epochs 10 --batch_size 1 --gradient_accumulation_steps 1
--lora_rank 32 --lora_alpha 64
--olora_mode standard
--lambda_fixed 1.0
--gamma 1.0
--tau 0.5                    # 本 baseline 不触发复用，但预留与 B3/Proposed 共用
--U_se_path <待确认路径>      # Task 0 投影
--output_dir general-knowledge/results/baselines/baseline2_standard_olora/run0
```

---

## 9. 非功能需求

- [x] **改代码 + 更新 README**（新 inner server + driver + shell）
- [ ] 单元测试：给定 dummy $A_t$ 与 $U$，ortho loss 梯度非零且 $\lambda=0$ 时 ortho 项关闭
- [ ] 与 Baseline 1 **同 split** 的集成测试（同 seed 前两篇 train/val 一致）
- [ ] log → summary 重建（可选）
- [ ] **公平性检查清单**：
  - [ ] 同 iter2、同 split、同 inner 容量（r/α/modules）
  - [ ] 同 merge 协议、同 eval 协议、同 GPT-4.1 判分
  - [ ] 仅差：ortho loss + $\mathcal{U}_{hist}$

---

## 10. 参考

- O-LoRA 原论文（正交低秩适应）
- SEAL `TTT_server.py` / `continual_self_edit_gen_forgetting.py`
- 项目 Adaptive O-LoRA 大纲 §4 Baseline 2

---

## 11. 实现架构（代码生成指南）

### 11.1 模块改动一览

```
baseline2_standard_olora.sh
    └── baseline2_standard_olora.py   # fork continual_self_edit_gen_forgetting
            ├── merge / eval 流程不变
            ├── 每步向 TTT_server_olora 传入 lambda_fixed=1.0, gamma, U_hist
            └── merge 后：extract A_t → append U_hist（train 步 only）

TTT_server_olora.py
    ├── 复用 TTT_server 数据管线与 eval
    ├── Trainer 自定义 loss：SFT + lambda * gamma * ortho(A, U_hist)
    └── 返回 adapter；可选返回 ortho 标量供日志

U_hist 存储
    ├── 内存：List[Dict[layer_name, Tensor]] 随 driver 持久化到
    │         output_dir/U_hist/merge_step_{k}.pt
    └── Task 0：load U_SE from --U_se_path
```

### 11.2 ZMQ 请求 schema（草案）

```json
{
  "train_sequences": ["..."],
  "eval_questions": [{"title","context","question","answer"}],
  "lora_rank": 32,
  "finetune_epochs": 10,
  "olora": {
    "enabled": true,
    "lambda_t": 1.0,
    "gamma": 1.0,
    "U_hist": "<serialized or path>",
    "ortho_on": "A",
    "accumulate_after_train": false
  }
}
```

### 11.3 与 Baseline 3 的共享组件（提前设计）

- `TTT_server_olora.py`：**共用** ortho loss 实现
- `U_hist` 读写工具：`general-knowledge/src/continual/olora_utils.py`（建议）
- $\tau$, $\gamma$, LoRA r/α **同默认值**

---

## 12. 待确认项（代码生成前）

| ID | 问题 | 默认决策 |
|----|------|----------|
| T1 | iter2 对应 **$W_{SE}$ adapter 路径** | 需用户确认；若无则方案 B |
| T2 | O-LoRA ortho 公式变体 | Frobenius $\sum_l \|A_t^{(l)} U^{(l)\top}\|_F^2$ |
| T3 | Val eval 是否使用 ortho | **是**（与 train 相同 inner），但 **不更新** $\mathcal{U}_{hist}$ |
| T4 | fork driver vs 扩展原脚本 | 推荐 **独立 baseline2*.py** 便于对照 |

---

## 13. 精简版

我要做 **Baseline 2 Standard O-LoRA**（**λ≡1.0**），目标是：在 iter2 起点 + **恒定强正交** 下，测 SE-gen forgetting 矩阵相对 Baseline 1 是否改善，并为「无自适应 O-LoRA 局限」提供对照。

**协议**：与 Baseline 1 相同 **(9×8)** single；每步 **新开** $W_{task_t}$；$\mathcal{U}_{hist}\leftarrow\{U_{SE}\}+\{A_t\}$；loss = SFT + **γ·ortho**。

**配置**：model=**models/iter2**，γ=**1.0**，λ=**1.0**，r=32，其余同 Baseline 1；OUTPUT=**baseline2_standard_olora/run0**。

**代码**：新 **TTT_server_olora** + **baseline2 driver**；Task 0 从 **$W_{SE}$ A 矩阵** 初始化 $\mathcal{U}_{hist}$。
