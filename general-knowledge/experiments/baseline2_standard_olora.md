# Baseline 2：Standard O-LoRA（恒定强正交，λ ≡ 1.0）

> 实验规格文档 · 用于代码生成与跑实验  
> 所属项目：Adaptive O-LoRA with Semantic Self-Reflection  
> 模板来源：`exp_template.md`  
> **Baseline 1 参照**：直接在 `continual_self_edit_gen_forgetting.sh` 上改配置完成（见 §0）

---

## 0. Baseline 1 已完成配置（本实验必须对齐）

Baseline 1 未单独建 shell，而是修改 `general-knowledge/scripts/continual_self_edit_gen_forgetting.sh` 后直接运行 vanilla driver。

### 0.1 Baseline 1 实际 shell 配置（`continual_self_edit_gen_forgetting.sh`）

| 变量 | Baseline 1 取值 |
|------|-----------------|
| `INNER_MODE` | `single` |
| `INDEX` | `0` |
| `MODEL_NAME` | `models/iter2` |
| `DATASET` | `general-knowledge/data/squad_val.json` |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_DROPOUT` | `32` / `64` / `0` |
| `FINETUNE_LR` | `1e-3` |
| `VLLM_SERVER_GPUS` / `PY_DRIVER_GPU` | `0` / `1` |
| `PORT` / `ZMQ_PORT` | `8001` / `5555`（`INDEX=0`） |
| `SEED` | `42`（`42 + INDEX`） |
| `MAX_TOKENS` / `TEMPERATURE` / `TOP_P` | `8192` / `1.0` / `0.95` |
| **`N_SEQUENCES`** | **`3`** |
| `N_MERGE` / `N_VAL` | `8` / `8` |
| `INNER_SFT_ARTICLES` | `1` |
| `K_COMP` | `1` |
| `FINETUNE_EPOCHS` / `BATCH_SIZE` / `GRAD_ACC` | `10` / `1` / `1` |
| `SPLIT_NEWLINES` | `1`（`--split_newlines`） |
| Inner 模式 | vanilla LoRA（无 ortho） |

### 0.2 Baseline 1 产出路径（勿覆盖）

| 产物 | 路径 |
|------|------|
| 结果目录 | `general-knowledge/results/continual_self_edit_gen_forgetting/single/run0` |
| Summary | `.../run0/summary_1782568260.json` |
| Split manifests | `.../run0/splits/seq{0,1,2}/{train,val,index}.json` |
| 热力图 | `.../run0/forgetting_heatmap_se_gen.png` |
| 矩阵规模 | **(9×8)**，`n_sequences=3`，`base_model=models/iter2` |

### 0.3 Baseline 2 与 Baseline 1 的唯一差分

| 项 | Baseline 1 | Baseline 2 |
|----|------------|------------|
| Inner loss | vanilla SFT | **SFT + γ·ortho**，**λ≡1.0** |
| $\mathcal{U}_{hist}$ | 无 | **有**：$\{U_{SE}\}$ + 每 merge 步追加 $A_t$ |
| Task LoRA 复用 | 无 | **无**（λ=1，恒新开 $W_{task}$） |
| $U_{SE}$ 来源 | 已 merge 进 iter2 | **`models/iter2_lora_adapter/`** |
| Driver / inner server | 现有 `continual_self_edit_gen_forgetting` + `TTT_server` | **新建** O-LoRA 版（见 §8） |

**其余全部相同**（model、dataset、seed、n_merge/n_val/n_sequences、inner 超参、eval 协议、GPU 布局）。

---

## 1. 实验目标（一句话 + 要回答的问题）

- **实验名称**：`baseline2_standard_olora`
- **核心问题**：在与 Baseline 1 **完全相同的 continual single 协议**下，对 inner LoRA 施加 **恒定满强度正交约束（$\lambda_t \equiv 1.0$）** 并维护 $\mathcal{U}_{hist}$，能否缓解 SE-gen forgetting？是否会暴露标准 O-LoRA 的局限（子空间枯竭、相似任务无法共享）？
- **Baseline 角色**：**Standard O-LoRA** 对照 —— 有正交、无相似度感知、无 LoRA 复用。
- **与已有实验的关系**：
  - [x] 与 Baseline 1 **同一 eval 矩阵** `(n_merge+1)×n_val`，**`n_sequences=3`**
  - [x] 与 Baseline 1 **同一起点** `models/iter2`
  - [x] Split 公平性：**`--seed 42`** 与 Baseline 1 相同（或显式复用 B1 的 `splits/`）
  - [ ] 代码：需 **O-LoRA inner**（`TTT_server_olora` + baseline2 driver）

---

## 2. 模式与矩阵协议

与 Baseline 1 **完全平行**（single passage only）。

| 维度 | 取值 |
|------|------|
| 每 task 文章数 | **1** |
| split_newlines | **True** |
| k_completions | **1** |
| inner epochs / bs / ga / lr | **10 / 1 / 1 / 1e-3** |
| $\lambda_t$ | **固定 1.0** |
| 任务 LoRA 复用 | **否** |
| 矩阵形状 | **(9×8)**：row 0 = 0 merge；row r = r 次 merge 后 |

- **Merge 训练**：每步 1 篇 train → fresh SE → **O-LoRA inner TTT**（λ=1）→ merge adapter。
- **Val 评估**：每个 checkpoint × 每个 val task → fresh SE + O-LoRA inner + adapter acc。
- **row 0**：仍做 inner SFT + ortho（$\mathcal{U}_{hist}=\{U_{SE}\}$），不是 zero-shot。

---

## 3. 数据与采样

- **数据集**：`general-knowledge/data/squad_val.json`
- **n_merge**：**8**
- **n_val**：**8**
- **n_sequences**：**3**（与 Baseline 1 及后续 Baseline 3/Proposed **统一**）
- **inner_sft_articles**：**1**
- **seed**：**42**（`SEED=$((42 + INDEX))`，`INDEX=0` → 42）
- **Split 公平性（硬约束）**：
  - 首选：driver 支持 `--splits_dir general-knowledge/results/continual_self_edit_gen_forgetting/single/run0/splits`，**直接复用 Baseline 1 的 seq0/1/2 split**，保证逐篇 train/val 完全一致。
  - 备选：同 `--seed 42` 重采样（应与 B1 一致，但复用 manifest 更稳妥）。
- **数据量**：$(8+8)\times1=16$ 篇/sequence，满足 dataset 规模。

---

## 4. 模型与 LoRA

### 4.1 生成器起点

- **Continual 起点**：`models/iter2`（与 Baseline 1 相同）

### 4.2 $W_{SE}$ / Task 0 投影（已确认，不再待填）

- **Outer adapter 路径（方案 A，已就绪）**：
  - 目录：`models/iter2_lora_adapter/`
  - A 矩阵：`models/iter2_lora_adapter/lora_A_matrices.pt`
  - 元信息：`models/iter2_lora_adapter/outer_lora_metadata.json`
- **验证过的配置**（与 inner 对齐）：
  - `lora_rank=32`, `lora_alpha=64`, `lora_dropout=0`
  - `lora_target_modules`: `q_proj`, `v_proj`
  - `n_lora_a_matrices`: **56**（28 层 × 2 modules）
- **$\mathcal{U}_{hist}$ 初始化**：$\mathcal{U}_{hist} \leftarrow \{U_{SE}\}$，从上述 `lora_A_matrices.pt` 加载。

### 4.3 Inner task LoRA（$W_{task_t}$）

与 Baseline 1 / `lora_config.py` **完全一致**：

| 参数 | 值 |
|------|-----|
| r | 32 |
| α | 64 |
| dropout | 0 |
| target_modules | `q_proj`, `v_proj` |
| 每 merge 步 | **新初始化** $W_{task_t}$（λ≡1） |

### 4.4 正交损失（Standard O-LoRA）

$$
\mathcal{L} = \mathcal{L}_{SFT}(P_t) + \lambda_t \cdot \gamma \cdot \mathcal{L}_{ortho}(A_t, \mathcal{U}_{hist}), \quad \lambda_t \equiv 1.0
$$

- **$\mathcal{L}_{ortho}$**：每层、每个 $U\in\mathcal{U}_{hist}$，Frobenius 惩罚 $\|A_t U^\top\|_F^2$（实现时与 O-LoRA 论文统一一种写法并写死）。
- **$\gamma$**：默认 **`1.0`**（Full 固定；summary 中记录）。
- **$\lambda_t$**：恒 **`1.0`**（`--lambda_fixed 1.0` / `--olora_mode standard`）。

### 4.5 $\mathcal{U}_{hist}$ 更新

| 时机 | 操作 |
|------|------|
| 实验开始 | 加载 $U_{SE}$ → $\mathcal{U}_{hist}=\{U_{SE}\}$ |
| 每 merge train 步完成 inner TTT | 将本步 $A_t$ **追加**到 $\mathcal{U}_{hist}$ |
| Val eval | 使用当前 checkpoint 的 $\mathcal{U}_{hist}$ 快照；**不向** bank 追加 |
| Merge | 与 B1 相同：完整 adapter `merge_and_unload` 进基座 |

---

## 5. 运行规模

- **n_sequences**：**3**（与 Baseline 1 一致；后续实验均按 3）
- **无单独 Smoke/Pilot 阶段**：直接 Full 配置（与 B1 相同规模）；若调试，可临时 `n_merge=2, n_val=2, n_sequences=1`。
- **GPU**：2 卡 — vLLM GPU `0`，inner GPU `1`（与 B1 shell 一致）。
- **端口**：`PORT=8001`, `ZMQ_PORT=5555`（`INDEX=0` 时；跑 B2 时勿与 B1 残留进程冲突）。

---

## 6. 产出物

- [ ] `summary_*.json`（字段含：`olora_mode=standard`, `gamma`, `lambda_fixed=1.0`, `U_se_path`, `U_hist_size_per_row`, `n_sequences=3`, `base_model=models/iter2`）
- [ ] `splits/seq*/`（推荐从 B1 复制或 `--splits_dir` 复用）
- [ ] `U_hist/merge_step_*.pt`（每 merge 步快照，便于调试）
- [ ] `forgetting_heatmap_se_gen.png`
- [ ] 可选：`forgetting_heatmaps_combined.png`（与 B1 并排对比，**不依赖** `continual_self_edits` inner 矩阵）
- [ ] O-LoRA 日志：每步 $\mathcal{L}_{SFT}$, $\mathcal{L}_{ortho}$；复用率恒 **0%**

### 6.1 输出目录（与 Baseline 1 分离）

```
general-knowledge/results/baselines/baseline2_standard_olora/run0
```

**禁止写入** `continual_self_edit_gen_forgetting/single/run0`（Baseline 1 结果）。

---

## 7. 指标定义

| 指标 | 定义 |
|------|------|
| **主矩阵 $(r,j)$** | checkpoint $r$ 上，val task $j$：**fresh SE + O-LoRA inner TTT → adapter acc** |
| **对比 Baseline 1** | 同 $(r,j)$ 格子比较 mean/std over **3 sequences** |
| **row 0** | iter2 + O-LoRA inner，$\mathcal{U}_{hist}=\{U_{SE}\}$ |
| **参数复用率** | **0%** |
| **Self-Edit 质量** | 后处理 Rouge-L / 格式率（与 B1 相同，非 driver 内置） |

---

## 8. 脚本与路径

### 8.1 入口脚本（新建，结构镜像 B1 shell）

**`general-knowledge/scripts/baseline2_standard_olora.sh`**

与 `continual_self_edit_gen_forgetting.sh` **相同区块**，仅增加 O-LoRA 变量并换 driver / OUTPUT：

```bash
INNER_MODE="single"          # 固定 single
INDEX=0
MODEL_NAME="models/iter2"
DATASET="general-knowledge/data/squad_val.json"

LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_LR=1e-3

VLLM_SERVER_GPUS="0"
PY_DRIVER_GPU="1"
PORT=$((8001 + INDEX))
ZMQ_PORT=$((5555 + INDEX))
SEED=$((42 + INDEX))

MAX_TOKENS=8192
TEMPERATURE=1.0
TOP_P=0.95

N_SEQUENCES=3
N_MERGE=8
N_VAL=8

INNER_SFT_ARTICLES=1
K_COMP=1
FINETUNE_EPOCHS=10
BATCH_SIZE=1
GRAD_ACC=1
SPLIT_NEWLINES=1

# --- O-LoRA only ---
OLORA_MODE="standard"
LAMBDA_FIXED=1.0
GAMMA=1.0
U_SE_PATH="models/iter2_lora_adapter/lora_A_matrices.pt"
# 可选：复用 B1 split
SPLITS_DIR="general-knowledge/results/continual_self_edit_gen_forgetting/single/run0/splits"

OUTPUT_DIR="general-knowledge/results/baselines/baseline2_standard_olora/run${INDEX}"
```

Python 入口（新建）：

```bash
python3 -u -m general-knowledge.src.continual.baseline2_standard_olora \
    ... # 与 B1 相同 CLI + 下列 O-LoRA 参数
    --olora_mode standard \
    --lambda_fixed 1.0 \
    --gamma 1.0 \
    --U_se_path "${U_SE_PATH}" \
    --splits_dir "${SPLITS_DIR}"   # 若 driver 已实现
```

绘图：

```bash
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
  --results_dir general-knowledge/results/baselines/baseline2_standard_olora/run0 \
  --output general-knowledge/results/baselines/baseline2_standard_olora/run0/forgetting_heatmap_se_gen.png
```

可选 combined（B1 vs B2）：

```bash
# 需扩展 plot 脚本或离线对比两个 summary 的 mean_over_sequences
```

### 8.2 代码模块（待实现）

| 模块 | 说明 |
|------|------|
| `src/continual/baseline2_standard_olora.py` | fork `continual_self_edit_gen_forgetting.py`；维护 $\mathcal{U}_{hist}$；调 O-LoRA inner |
| `src/inner/TTT_server_olora.py` | fork `TTT_server.py`；loss = SFT + λ·γ·ortho |
| `src/continual/olora_utils.py` | 加载/追加 `U_hist`；读写 A 矩阵；与 `lora_checkpoint_utils` 衔接 |

### 8.3 ZMQ 请求扩展（草案）

```json
{
  "train_sequences": ["..."],
  "eval_questions": [{"title","context","question","answer"}],
  "lora_rank": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  "finetune_epochs": 10,
  "finetune_lr": 0.001,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "end_mask_substring": "",
  "olora": {
    "enabled": true,
    "lambda_t": 1.0,
    "gamma": 1.0,
    "U_hist_path": "…/U_hist/merge_step_k.pt"
  }
}
```

---

## 9. 公平性检查清单（跑 B2 前）

- [ ] `MODEL_NAME=models/iter2`
- [ ] `N_SEQUENCES=3`, `N_MERGE=8`, `N_VAL=8`, `SEED=42`
- [ ] Inner：r=32, α=64, q/v, epochs=10, bs=1, ga=1, lr=1e-3, split_newlines=True, k=1
- [ ] `U_SE_PATH=models/iter2_lora_adapter/lora_A_matrices.pt` 存在且 rank=32
- [ ] Split 与 B1 一致（`splits_dir` 或同 seed）
- [ ] OUTPUT 在 `baselines/baseline2_standard_olora/run0`
- [ ] 仅差 ortho + $\mathcal{U}_{hist}$，其余与 B1 相同

---

## 10. 与 Baseline 3 / Proposed 的统一约定

以下参数 **三者在 Full 实验中保持一致**（已在 B1 跑通）：

```
model=models/iter2
dataset=squad_val.json
seed=42, INDEX=0
n_sequences=3, n_merge=8, n_val=8
inner: r=32, α=64, q/v, epochs=10, bs=1, ga=1, lr=1e-3, k=1, split_newlines=True
SE: max_tokens=8192, temperature=1.0, top_p=0.95
U_SE: models/iter2_lora_adapter/
```

| 实验 | 额外差分 |
|------|----------|
| B2 | λ≡1, γ=1, 无复用 |
| B3 | embedding λ, 可复用 |
| Proposed | LLM Step3 λ, 可复用 |

---

## 11. 待实现项（代码生成任务）

| ID | 任务 | 优先级 |
|----|------|--------|
| C1 | `TTT_server_olora.py` + ortho loss | P0 |
| C2 | `baseline2_standard_olora.py` + `U_hist` 生命周期 | P0 |
| C3 | `baseline2_standard_olora.sh`（§8.1 配置） | P0 |
| C4 | `--splits_dir` 复用 B1 splits | P1 |
| C5 | summary 增加 O-LoRA 字段 | P1 |
| C6 | B1 vs B2 对比 plot | P2 |

---

## 12. 精简版（Agent 开干用）

我要做 **Baseline 2 Standard O-LoRA**，与 **Baseline 1 完全同协议**，仅加 ortho。

**B1 参照**：`continual_self_edit_gen_forgetting.sh` + `.../single/run0`（vanilla，3 seq，iter2，9×8 矩阵已完成）。

**配置（与 B1 shell 一致 + O-LoRA）**：
`model=models/iter2`，`dataset=squad_val.json`，`seed=42`，`n_sequences=3`，`n_merge=8`，`n_val=8`，inner `10/1/1/1e-3`，`k=1`，`split_newlines`，`r=32, α=64, q/v`。

**O-LoRA**：`λ=1.0`，`γ=1.0`，`U_SE=models/iter2_lora_adapter/lora_A_matrices.pt`，每 merge 追加 $A_t$，无复用。

**产出**：`general-knowledge/results/baselines/baseline2_standard_olora/run0` + summary + 热力图。

**Split**：复用 `continual_self_edit_gen_forgetting/single/run0/splits`。

**代码**：`baseline2_standard_olora.py` + `TTT_server_olora.py` + `baseline2_standard_olora.sh`。
