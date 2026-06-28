# Baseline 3：Embedding-based Adaptive O-LoRA（BGE 相似度控权）

> 实验规格文档 · 用于下一步代码生成  
> 所属项目：Adaptive O-LoRA with Semantic Self-Reflection  
> 模板来源：`exp_template.md`

---

## 1. 实验目标（一句话 + 要回答的问题）

- **实验名称**：`baseline3_embedding_adaptive_olora`
- **核心问题**：在 continual single-passage 场景下，用 **外部 Embedding（如 BGE）** 估计任务相似度并 **自适应调节正交系数 $\lambda_t$**，以及在 **$\lambda_t<\tau$ 时复用历史 task LoRA**，能否优于 Baseline 1/2？该 baseline 用于证明：后续 Proposed 方法中 **LLM 语义自省（Step 3）** 相对 **向量空间几何距离** 的增量价值。
- **Baseline 角色**：公平对照矩阵中的 **最强非 LLM-introspection 对手** —— 有 RAG + 自适应 $\lambda_t$ + 参数复用，但 **$\lambda_t$ 来自 embedding 余弦相似度**，**非** LLM Step 3 自省。
- **与已有实验的关系**：
  - [x] 与 Baseline 1/2 **同一 eval 矩阵协议**、**同一 `models/iter2` 起点**
  - [x] 与 Baseline 2 **共用 O-LoRA inner loss 实现**（`TTT_server_olora`）
  - [x] 与 Proposed 方法 **流程同构**（两步法 RAG + 自适应 loss + 复用），仅 **Step 3 相似度来源不同**
  - [ ] 与 `continual_self_edit_gen_forgetting`：需 **新 task bank + retrieval driver**

### 1.1 与用户补充说明的对齐

- 研究大纲 §4 定义本 baseline：**BGE 等 Embedding 余弦相似度 → $\lambda_t$**（**不用** LLM Step 3 自省）。
- 若实现阶段暂用 GPT-4.1 打相似度分，应记为 **ablation**（`baseline3_gpt4_similarity_ablation`），**不得**替代本文件主结果，以免与 Proposed 混淆。

---

## 2. 模式与「single / CPT 对称性」

与 Baseline 1 **eval 协议平行**；与 Proposed **训练决策流平行**。

| 维度 | Baseline 3 | Baseline 2 | Proposed |
|------|------------|------------|----------|
| 相似度来源 | **BGE embedding** | 无（λ=1） | **LLM Step 3 自省** |
| $\lambda_t$ | **自适应** $[0,1]$ | 固定 1.0 | 自适应 $[0,1]$ |
| Top-K 检索 | **是**（Step 2） | 否 | 是 |
| $\lambda<\tau$ 复用 $W_{task}$ | **是** | 否 | 是 |
| Task bank | **是**（$I_t$ + $A_t$） | 仅 $\mathcal{U}_{hist}$ | 是 + 更丰富元数据 |

- **Merge 训练（每个 train 时间步 $t$）**：
  1. **Step 1 知识生成**：$I_t = \text{LLM}(P_t; \Theta_{Base}^{(t-1)})$（iter2 演化后的当前 merged 模型；**不**在 inner 中更新 $W_{SE}$ meta 头）。
  2. **Step 2 检索**：$\mathcal{Task}_{hist} = \text{Retrieve}(I_t, \mathcal{K}, K)$；Query 为 **$I_t$ 拼接成的 passage 表示**（见 §4.3）。
  3. **Step 3 控权（Embedding 版）**：$\lambda_t = f(\text{sim}_{BGE}(I_t, I_{hist}))$，**非 LLM 生成**。
  4. **决策 + Inner O-LoRA SFT**（§4.4）→ merge。
- **Val 评估**：每个 checkpoint、每个 val task：**重复 Step 1–4**（fresh $I$、fresh retrieval、fresh $\lambda$）；**不写入** train task bank。
- **矩阵形状**：`(n_merge + 1) × n_val`。

### 2.1 冷启动与 $t=1$ 硬拦截（与方法论一致）

- **$\mathcal{K}$（task bank）**：初始 **空**（不含 train 条目；$U_{SE}$ 仅进入 $\mathcal{U}_{hist}$，**不强制**进入 retrieval bank，除非检索实现将 Task 0 作为虚拟条目 —— **默认：bank 仅含已完成 merge 的真实 train 任务**）。
- **Merge 步 $t=1$（首个 train 文章）**：**硬编码** $\lambda_1=1.0$，**不复用**；新开 $W_{task_1}$。
- **Merge 步 $t\ge 2$**：启用 embedding 相似度 + 复用逻辑。
- **Val row 0（0 merge）**：对 val 任务 **不依赖 train bank**；$\lambda=1.0$（与 $t=1$ 同逻辑：无历史 train 轨迹）。

---

## 3. 数据与采样

- **数据集路径**：`general-knowledge/data/squad_val.json`
- **n_merge / n_val / inner_sft_articles**：**8 / 8 / 1**（Full）；Smoke：**2 / 2 / 1**
- **去重与 split**：**与 Baseline 1/2 相同 seed、相同 split**（强制）。
- **数据量检查**：16 篇（Full）。

---

## 4. 模型、Task Bank 与 O-LoRA

### 4.1 基座

- **起点**：`models/iter2`（含 RL 对齐的 SE 能力）。
- **$W_{SE}$**：已 merge 进 iter2；Continual 过程中 **仅通过 merged 全量权重** 体现，**不在 inner loop 单独训练** meta LoRA。

### 4.2 Task Bank $\mathcal{K}$ 条目结构

每完成一个 **merge train 步** $t$，向 $\mathcal{K}$ 追加一条记录：

```json
{
  "task_id": "merge_t_{seq}_{k}",
  "article_key": {"title": "...", "context": "..."},
  "I_t_text": "<Step1 生成的 implication 列表拼接 passage>",
  "embedding": "<float32[BGE_DIM] 可选缓存>",
  "lora_adapter_path": "<该步 inner 训完、merge 前或 merge 后快照路径>",
  "A_matrices": "<Dict[layer, Tensor] 该 task LoRA 的 A，用于 U_hist 与复用>",
  "merge_step": k,
  "seq_idx": s
}
```

- **存储路径**：`{output_dir}/task_bank/seq{s}/task_{k}.json` + `.pt`（A 矩阵 / adapter）。
- **$I_t$ 文本构造（统一规范）**：
  - 使用 Step 1 原始 completion $I_t$；
  - 若含 `Implications:` / `---` 分段，**保留生成正文**（与 `build_train_sequences` 前处理一致）；
  - 拼接为 **单段文本**（换行保留），**不**拼接原始 $P_t$ context（避免检索退化为 passage 重复）。

### 4.3 Step 2：Embedding 检索

- **Embedding 模型**：**BAAI/bge-large-en-v1.5**（默认；可配置 `--embed_model`）。
- **编码对象**：Query = $I_t$ 文本；Key = 各历史 $I_{hist}$ 文本。
- **相似度**：余弦相似度 $\text{sim}_i = \cos(e(I_t), e(I_i))$。
- **Top-K**：**K=3**（默认 `--retrieve_k 3`）；不足 K 时取全部。
- **输出**：$\mathcal{Task}_{hist}$ = Top-K 条目（供日志与可选 prompt 上下文；**控权只用相似度标量**）。

### 4.4 Step 3：$\lambda_t$ 映射（Embedding → $[0,1]$）

**不让 LLM 输出 $\lambda_t$**；由 embedding 相似度确定性映射：

- 令 $s_{\max} = \max_{i \in \text{Top-K}} \text{sim}_i$（若无历史，$\lambda=1$）。
- **默认映射（推荐）**：
  $$
  \lambda_t = \mathrm{clip}(1 - s_{\max},\ 0,\ 1)
  $$
  - 解释：越相似（$s_{\max}\to 1$）→ $\lambda_t\to 0$（弱正交、允许复用子空间）；越不相似 → $\lambda_t\to 1$。
- **可选映射**（ablation）：$\lambda_t = \mathrm{clip}(1 - \frac{s_{\max}+1}{2}, 0, 1)$ 等 —— Full 实验 **固定一种** 并写入 summary。

**强相关阈值 $\tau$**：**默认 `0.5`**（与 Proposed 共用 CLI `--tau`）。

### 4.5 Step 4：决策分流与 Loss

| 条件 | 行为 |
|------|------|
| $\lambda_t < \tau$ | **复用** Top-1 最相似历史的 **task LoRA 初始化**（加载 $W_{task_{hist}}$ 作为 warm-start），**不新开 rank**；正交 loss 仍对 **当前 A** 与 $\mathcal{U}_{hist}$ 计算，系数 **$\lambda_t \cdot \gamma$** |
| $\lambda_t \ge \tau$ | **新初始化** $W_{task_t}$；$\mathcal{L}_{ortho}$ 系数 **$\lambda_t \cdot \gamma$**（通常 $\approx \gamma$） |

$$
\mathcal{L} = \mathcal{L}_{SFT}(P_t) + \lambda_t \cdot \gamma \cdot \mathcal{L}_{ortho}(A, \mathcal{U}_{hist})
$$

- **$\gamma$**：默认 **1.0**（与 Baseline 2 相同）。
- **$\mathcal{U}_{hist}$**：
  - 初始化：$\{U_{SE}\}$（同 Baseline 2 §4.1）。
  - 更新：仅当 **新开** $W_{task_t}$ 且完成 train 后，将 **本 task 的 A** 追加（**复用步是否追加**：**否** —— 不重复计数同一子空间；若实现困难，**是** 但不追加 duplicate task_id）。

### 4.6 Merge 与 Bank 一致性

- **Merge 协议**：与 Baseline 1 相同（adapter merge 进基座）。
- **复用步 merge**：merge 的是 **在本 task data 上继续 SFT 后的 adapter**（起点为历史 LoRA）。
- **Inner 超参**：r=32, α=64, epochs=10, bs=1, ga=1, split_newlines=True, k=1。

---

## 5. 运行规模与优先级

- **n_sequences**：**5**（Full）；Smoke：**1**
- **阶段计划**：
  - [x] **Smoke**：2 merge + 2 val；验证 bank 写入、$t=1$ 拦截、复用触发（人工选相似 passage 可选）
  - [x] **Full**：8+8，5 seq
- **依赖**：BGE 模型下载；CPU/GPU embedding 批推理（**不占用 inner GPU** 为宜）。
- **超时砍序**：n_sequences → n_merge → retrieve_k

---

## 6. 产出物

- [x] `summary_*.json`（含 `lambda_mean_per_row`, `reuse_rate`, `tau`, `embed_model`, `lambda_mapping`）
- [x] `task_bank/seq*/` 完整轨迹
- [x] 热力图：`forgetting_heatmap_se_gen.png`
- [x] **Adaptive 专用 CSV/JSONL**（建议）：
  - 每 merge 步：$s_{\max}$, $\lambda_t$, 是否复用, matched_task_id
  - **参数空间复用率** = 复用步数 / 总 merge 步数（按 seq 平均）
- [ ] Rouge-L / 格式率：后处理 Step 1 的 $I_t$

---

## 7. 指标定义

| 指标 | 定义 |
|------|------|
| **主矩阵 $(r,j)$** | 同 Baseline 1；inner = **Adaptive O-LoRA**（embedding $\lambda_t$） |
| **row 0** | iter2 + fresh SE + adaptive inner（无 train bank → $\lambda=1$） |
| **下游 QA** | adapter accuracy |
| **Self-Edit 质量** | Rouge-L($I_t$, $P_t$) + 格式率（`Implications:` / 列表项比例） |
| **参数空间复用率** | $\#\{\lambda_t<\tau\}/n_{merge}$（per seq，再平均） |
| **$\lambda_t$ 分布** | 每 row/checkpoint 上 merge 步的均值与直方图 |

---

## 8. 脚本与路径

- **入口脚本（新建）**：`general-knowledge/scripts/baseline3_embedding_adaptive_olora.sh`
- **Python driver（新建）**：`general-knowledge/src/continual/baseline3_embedding_adaptive_olora.py`
- **共用模块（与 B2 共建）**：
  - `general-knowledge/src/inner/TTT_server_olora.py`
  - `general-knowledge/src/continual/olora_utils.py`（$\mathcal{U}_{hist}$, A 提取, adapter 加载）
  - `general-knowledge/src/continual/task_bank.py`（$\mathcal{K}$ CRUD + embedding 检索）
- **OUTPUT_DIR**：`general-knowledge/results/baselines/baseline3_embedding_adaptive_olora/run${INDEX}`

### 8.1 建议 CLI

```
--model models/iter2
--dataset general-knowledge/data/squad_val.json
--n_merge 8 --n_val 8 --n_sequences 5 --seed 42
--inner_sft_articles 1 --k_completions 1 --split_newlines
--finetune_epochs 10 --batch_size 1 --gradient_accumulation_steps 1
--lora_rank 32 --lora_alpha 64
--olora_mode adaptive
--lambda_source embedding
--embed_model BAAI/bge-large-en-v1.5
--retrieve_k 3
--tau 0.5
--gamma 1.0
--lambda_mapping "1_minus_max_sim"
--U_se_path <同 Baseline 2>
--output_dir general-knowledge/results/baselines/baseline3_embedding_adaptive_olora/run0
```

---

## 9. 非功能需求

- [x] **改代码 + 更新 README**
- [x] 与 Baseline 2 **共用 ortho 内核**；与 Baseline 1 **共用 eval 矩阵与 plot 脚本**（扩展 summary 字段）
- [ ] Embedding 缓存：同 $I_t$ 文本不重复编码
- [ ] 公平性清单：
  - [ ] 同 iter2、同 split、同 inner 容量、同 merge/eval 协议
  - [ ] 与 Baseline 2 **差分仅**：λ 来源 + 复用 + bank
  - [ ] 与 Proposed **差分仅**：Step 3 用 BGE 而非 LLM 自省

---

## 10. 参考

- BGE 系列 embedding 模型
- O-LoRA + SEAL 代码路径（见 Baseline 2）
- Adaptive O-LoRA 大纲 §4 Baseline 3

---

## 11. 实现架构（代码生成指南）

### 11.1 主循环（merge 步 $k$）

```
1. Step1: vLLM 生成 I_t (make_prompt + completion)
2. if k==0 (first merge step): lambda=1.0, reuse=False
   else:
     Step2: Task_hist = retrieve(I_t, K, bank)
     Step3: lambda = map(sim_max)  # embedding
     reuse = (lambda < tau) → load best_match adapter init
3. Build train_sequences from I_t (split_newlines=True)
4. ZMQ → TTT_server_olora:
     { lambda_t, gamma, U_hist, init_adapter_path? }
5. merge adapter → update base model
6. if not reuse: append A to U_hist
7. append task record to bank (I_t, embedding, adapter ref, A)
8. eval all val tasks (each: Step1-4, no bank write)
```

### 11.2 Val eval 子流程

- 对每个 val 文章 $P_j$：**独立** Step 1–4（检索 **train bank 当前快照**，不含 val 自身）。
- 首个 checkpoint row 0：**bank 空** → $\lambda=1$。

### 11.3 ZMQ 扩展字段（在 Baseline 2 基础上）

```json
{
  "olora": {
    "enabled": true,
    "lambda_t": 0.32,
    "gamma": 1.0,
    "U_hist": "...",
    "init_adapter_path": "optional when reuse",
    "reuse_mode": true
  }
}
```

### 11.4 绘图

- 复用 `plot_self_edit_gen_forgetting.py` 画 **QA 矩阵**。
- 新增 `plot_adaptive_olora_diagnostics.py`（可选）：$\lambda_t$ 曲线、复用率柱状图。

---

## 12. 四组 Baseline 公平对照表（Master Config）

| 项 | B1 Vanilla | B2 Standard O-LoRA | B3 Embedding Adaptive | Proposed（后续） |
|----|------------|--------------------|-----------------------|------------------|
| 初始模型 | models/iter2 | iter2 | iter2 | iter2 |
| split/seed | 42, 8+8 | **同** | **同** | **同** |
| inner r/α | 32/64 | 32/64 | 32/64 | 32/64 |
| epochs/bs/ga | 10/1/1 | 10/1/1 | 10/1/1 | 10/1/1 |
| ortho | 无 | γ, λ=1 | γ, λ_t(BGE) | γ, λ_t(LLM) |
| U_SE init | 无 | 有 | 有 | 有 |
| task bank | 无 | 无 | 有 | 有 |
| LoRA 复用 | 无 | 无 | λ<τ | λ<τ |
| Step3 | 无 | 无 | BGE | LLM 自省 |

---

## 13. 待确认项

| ID | 问题 | 建议 |
|----|------|------|
| T1 | $W_{SE}$ / $U_{SE}$ 路径 | 与 Baseline 2 共用结论 |
| T2 | 复用步是否 merge **增量 adapter** 还是 **全量** | 与 SEAL 一致：本步 SFT 后的 adapter merge |
| T3 | BGE 运行设备 | 默认 CPU；batch encode |
| T4 | $\tau=0.5$ 是否做 pilot 扫描 | Full 固定 0.5；appendix 扫 {0.3,0.5,0.7} |

---

## 14. 精简版

我要做 **Baseline 3 Embedding Adaptive O-LoRA**：iter2 起点；**Task bank** 存 ($I_t$, $A_t$)；新任务 **BGE Top-K 检索** → $\lambda_t=1-\max\cos$ → 若 $\lambda<\tau$ **复用**历史 LoRA，否则新开；loss = SFT + **λ_t·γ·ortho**；$t=1$ **强制 λ=1**。

**协议**：single **(9×8)** 矩阵同 B1；val 每格 fresh SE+RAG+inner。

**配置**：BGE=**bge-large-en-v1.5**，K=**3**，τ=**0.5**，γ=**1.0**；其余 inner 同 B1。

**产出**：`baseline3_embedding_adaptive_olora/run0` + bank + **复用率** + QA 热力图。

**注意**：λ **必须来自 embedding**；GPT-4.1 打分仅作 ablation，非本 baseline。
