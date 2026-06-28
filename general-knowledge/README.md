# SEAL - general-knowledge

This is an implementation of SEAL for the *general knowledge incorporation* setting, where the goal is to update or integrate new information from a passage into weights.

## Usage

The python files in src/ have documentation on function. Here is some information on how to run the pipelines used in the paper's experiments.

### 1. Create Data
Use `make_squad_data.sh` (or `make_squad_data_openai.sh`) to create the synthetic data used in subsequent RL training or evaluation.

```bash
sbatch general-knowledge/scripts/make_squad_data.sh
```

### 2. TTT server
Run the `TTT_server`. This sets up a [ZMQ](https://zeromq.org/) port that takes input parameters like training data and corresponding questions, and then runs rounds of training a temporary lora adapter and evaluating on the questions. This is then called for both RL training rewards and evaluation.

```bash
sbatch general-knowledge/scripts/TTT_server.sh
```

### 3. Query server
To query the server, run either `query_server` or `CPT` for either the single-passage or multi-passage setting respectively. This can be set to run on training documents for a round of ReST-EM RL training, or on validation documents for evaluation. 

```bash
sbatch general-knowledge/scripts/query_server.sh
```

### 4. RL Training
To run a round of ReST-EM, after running `query_server` on training documents, build the SFT dataset (more documentation in the python file):

```bash
python3 general-knowledge/src/EM/build_SFT_dataset.py <path/to/result/of/run.json>
```

Then, run the training script on this dataset:

```bash
sbatch general-knowledge/scripts/train_SFT.sh
```

### 5. Continual Self-Edits
To run the continual self-edits experiment (Section 5):

```bash
sbatch general-knowledge/scripts/continual_self_edits.sh
```

### 5b. Self-Edit Generation Forgetting (Experiment 1)
Continual **merge** on `n_merge` train passages. After each merge checkpoint, fresh SE + inner TTT + eval.

- `--inner_sft_articles 1` (**single**): `n_merge` / `n_val` disjoint **1-article** tasks; per-passage acc; `split_newlines=True`.
- `--inner_sft_articles 200` (**CPT**): same layout — `n_merge` train tasks + `n_val` val tasks, each **200 articles**; fresh SE + aggregated inner TTT per task; val = overall acc (`k_completions=5`, `epochs=3`, `batch_size=4`, `grad_acc=2`, `split_newlines=False`, matching `CPT.sh`). Matrix `(n_merge+1) × n_val`. Needs `(n_merge+n_val)×200` unique articles in the dataset.

```bash
# In continual_self_edit_gen_forgetting.sh set INNER_MODE=single or INNER_MODE=cpt
sbatch general-knowledge/scripts/continual_self_edit_gen_forgetting.sh
```

Train/val splits per sequence: `results/.../splits/seq*/{train,val}.json`. Plot:

```bash
# SE-gen full matrix only
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
  --results_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0

# SE-gen (full) + continual_self_edits inner-TTT (lower-triangular)
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
  --results_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0 \
  --inner_results_dir general-knowledge/results/continual_self_edits/run0 \
  --combined_output general-knowledge/results/continual_self_edit_gen_forgetting/run0/forgetting_heatmaps_combined.png
```
