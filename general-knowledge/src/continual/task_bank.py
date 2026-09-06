# general-knowledge/src/continual/task_bank.py
"""Task bank CRUD, U_hist anchor mapping, and per-task similarity lambdas."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .olora_utils import MANIFEST_NAME


@dataclass
class TaskBankEntry:
    task_id: str
    article_key: Dict[str, str]
    I_t_text: str
    lora_adapter_path: str
    A_matrices_path: Optional[str]
    merge_step: int
    seq_idx: int
    lambda_t_at_train: float
    similarity_source: str
    s_max_at_train: float
    reused_from: Optional[str]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "article_key": self.article_key,
            "I_t_text": self.I_t_text,
            "embedding": self.embedding,
            "lora_adapter_path": self.lora_adapter_path,
            "A_matrices_path": self.A_matrices_path,
            "merge_step": self.merge_step,
            "seq_idx": self.seq_idx,
            "lambda_t_at_train": self.lambda_t_at_train,
            "similarity_source": self.similarity_source,
            "s_max_at_train": self.s_max_at_train,
            "reused_from": self.reused_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskBankEntry":
        return cls(
            task_id=data["task_id"],
            article_key=data["article_key"],
            I_t_text=data["I_t_text"],
            lora_adapter_path=data["lora_adapter_path"],
            A_matrices_path=data.get("A_matrices_path"),
            merge_step=int(data["merge_step"]),
            seq_idx=int(data["seq_idx"]),
            lambda_t_at_train=float(data.get("lambda_t_at_train", 1.0)),
            similarity_source=data.get("similarity_source", "gpt4"),
            s_max_at_train=float(data.get("s_max_at_train", 0.0)),
            reused_from=data.get("reused_from"),
            embedding=data.get("embedding"),
        )


class BGEEncoder:
    """Lazy BGE encoder on CPU for embedding-based similarity."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        assert self._tokenizer is not None and self._model is not None
        if is_query:
            texts = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self._model(**encoded)
        token_embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class TaskBank:
    """On-disk task bank for one sequence."""

    def __init__(
        self,
        bank_dir: Path,
        seq_idx: int,
        *,
        embed_model: str = "BAAI/bge-large-en-v1.5",
    ):
        self.bank_dir = Path(bank_dir)
        self.seq_idx = seq_idx
        self.embed_model = embed_model
        self.adapters_dir = self.bank_dir / "adapters"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self._encoder: Optional[BGEEncoder] = None

    @property
    def encoder(self) -> BGEEncoder:
        if self._encoder is None:
            self._encoder = BGEEncoder(self.embed_model, device="cpu")
        return self._encoder

    def _entry_path(self, merge_step: int) -> Path:
        return self.bank_dir / f"task_{merge_step:03d}.json"

    def is_empty(self) -> bool:
        return len(list(self.bank_dir.glob("task_*.json"))) == 0

    def list_entries(self) -> List[TaskBankEntry]:
        entries: List[TaskBankEntry] = []
        for path in sorted(self.bank_dir.glob("task_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.append(TaskBankEntry.from_dict(data))
        return entries

    def get_entry(self, task_id: str) -> Optional[TaskBankEntry]:
        for entry in self.list_entries():
            if entry.task_id == task_id:
                return entry
        return None

    def append_entry(
        self,
        *,
        merge_step: int,
        item: Dict[str, Any],
        i_t_text: str,
        adapter_src: Path,
        a_matrices_path: Optional[Path],
        lambda_t: float,
        s_max: float,
        similarity_source: str,
        reused_from: Optional[str] = None,
    ) -> TaskBankEntry:
        task_id = f"merge_s{self.seq_idx}_k{merge_step}"
        adapter_dst = self.adapters_dir / f"task_{merge_step:03d}"
        if adapter_dst.exists():
            shutil.rmtree(adapter_dst)
        shutil.copytree(adapter_src, adapter_dst)

        embedding = self.encoder.encode([i_t_text], is_query=False)[0].tolist()
        entry = TaskBankEntry(
            task_id=task_id,
            article_key={"title": item["title"], "context": item["context"]},
            I_t_text=i_t_text,
            lora_adapter_path=str(adapter_dst.resolve()),
            A_matrices_path=str(a_matrices_path.resolve()) if a_matrices_path else None,
            merge_step=merge_step,
            seq_idx=self.seq_idx,
            lambda_t_at_train=lambda_t,
            similarity_source=similarity_source,
            s_max_at_train=s_max,
            reused_from=reused_from,
            embedding=embedding,
        )
        self._entry_path(merge_step).write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return entry


def lambda_from_similarity(s_max: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(s_max)))


def u_hist_anchor_entries(
    task_bank: "TaskBank",
    u_hist_dir: Path,
) -> List[TaskBankEntry]:
    """
    Bank entries whose A matrix is in U_hist, ordered to match u_hist[1:].
    """
    manifest_path = Path(u_hist_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merge_tasks: List[str] = list(manifest.get("merge_tasks") or [])
    if not merge_tasks:
        return []

    by_a_path: Dict[str, TaskBankEntry] = {}
    for entry in task_bank.list_entries():
        if entry.A_matrices_path:
            by_a_path[Path(entry.A_matrices_path).name] = entry

    anchors: List[TaskBankEntry] = []
    for fname in merge_tasks:
        entry = by_a_path.get(fname)
        if entry is not None:
            anchors.append(entry)
    return anchors


def _embedding_similarity(
    query_i_t_text: str,
    entry: TaskBankEntry,
    encoder: BGEEncoder,
) -> float:
    query_emb = encoder.encode([query_i_t_text], is_query=True)[0]
    if entry.embedding:
        key_emb = np.array(entry.embedding, dtype=np.float32)
    else:
        key_emb = encoder.encode([entry.I_t_text], is_query=False)[0]
    return _cosine_sim(query_emb, key_emb)


def compute_per_uhist_lambdas_embedding(
    query_i_t_text: str,
    anchors: List[TaskBankEntry],
    encoder: BGEEncoder,
) -> Dict[str, float]:
    """Per U_hist anchor task: lambda_i = 1 - cosine_sim."""
    return {
        entry.task_id: lambda_from_similarity(
            _embedding_similarity(query_i_t_text, entry, encoder)
        )
        for entry in anchors
    }


def compute_per_uhist_similarities_embedding(
    query_i_t_text: str,
    anchors: List[TaskBankEntry],
    encoder: BGEEncoder,
) -> Dict[str, float]:
    """Per U_hist anchor task: raw cosine similarity in [0, 1] (typical range)."""
    return {
        entry.task_id: _embedding_similarity(query_i_t_text, entry, encoder)
        for entry in anchors
    }


def build_adaptive_decision_from_anchor_lambdas(
    *,
    anchor_lambdas: Dict[str, float],
    anchors: List[TaskBankEntry],
    task_bank: "TaskBank",
    tau: float,
    u_se_lambda: float = 1.0,
) -> Tuple[List[float], float, float, bool, Optional[str], Optional[str]]:
    """
    Build lambda_weights aligned with u_hist [U_SE, anchor_1, ...] and reuse decision.

    Reuse when min(anchor lambda) < tau; warm-start from that anchor's bank adapter.
    """
    lambda_weights = [u_se_lambda]
    per_task = dict(anchor_lambdas)

    for entry in anchors:
        lambda_weights.append(per_task.get(entry.task_id, 1.0))

    if not anchor_lambdas:
        return lambda_weights, 1.0, 0.0, False, None, None

    matched_id = min(anchor_lambdas, key=anchor_lambdas.get)
    min_lambda = anchor_lambdas[matched_id]
    s_max = 1.0 - min_lambda
    reuse = min_lambda < tau
    init_path = None
    if reuse:
        matched = task_bank.get_entry(matched_id)
        if matched:
            init_path = matched.lora_adapter_path
        else:
            reuse = False
            matched_id = None
    return (
        lambda_weights,
        min_lambda,
        s_max,
        reuse,
        matched_id if reuse else None,
        init_path,
    )
