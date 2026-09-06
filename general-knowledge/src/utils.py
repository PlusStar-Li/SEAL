# general-knowledge/src/utils.py
import json
import requests
import logging
import time
from typing import Any, Dict, List, Optional
import re
from openai import OpenAI

# ---------------------------  CONFIG  ---------------------------------- #
# Qwen-base answering template
SQUAD_ANSWER_TEMPLATE_BASE = (
    "Let's answer a question directly and concisely.\n"
    "Question: {question}\n"
    "Answer:\n"
)

# Qwen-base answering template with chain of thought
SQUAD_ANSWER_TEMPLATE_BASE_COT = (
    "Let's think step by step and then answer the question directly and concisely. "
    "Let's first give reasoning under \"Reasoning:\" and then the final answer under \"Final answer:\".\n"
    "Question: {question}\n"
    "Reasoning:"
)

# Qwen-instruct answering
SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant to answer a question directly and concisely."
    "<|im_end|>\n"
    "<|im_start|>user\n{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Qwen3 chat template: empty think block == enable_thinking=false (non-thinking).
QWEN3_DISABLE_THINKING_PREFIX = "<think>\n\n</think>\n\n"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from model output."""
    return _THINK_RE.sub("", text or "").strip()


def apply_qwen3_thinking_prefix(prompt: str, *, instruct_model: bool, thinking_mode: bool) -> str:
    """
    Align a chat-template prompt with Qwen3 thinking on/off used at inference.

    - instruct + thinking_mode: ensure no empty <think></think> prefix (model may reason)
    - instruct + not thinking_mode: append empty <think></think> (enable_thinking=false)
    - not instruct: leave prompt unchanged
    """
    if not instruct_model:
        return prompt
    # Drop a trailing empty think block if present, then re-apply policy.
    if prompt.endswith(QWEN3_DISABLE_THINKING_PREFIX):
        prompt = prompt[: -len(QWEN3_DISABLE_THINKING_PREFIX)]
    if not thinking_mode:
        prompt = prompt + QWEN3_DISABLE_THINKING_PREFIX
    return prompt

# Qwen grading (unused)
SQUAD_GRADE_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are a grading assistant. Your job is to determine whether a student's answer "
    "correctly answers the question based solely on the provided gold answer. Do not use any outside knowledge. "
    "The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>user\n{question}\n"
    "Gold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# OpenAI grading
SQUAD_GRADE_TEMPLATE = (
    "You are a grading assistant. Your job is to determine whether a student's answer correctly answers the question based solely on the provided gold answer. "
    "Do not use any outside knowledge. The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)

TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"
# ----------------------------------------------------------------------- #

# vLLM API thin wrapper
API = requests.Session()
VLLM_API_URL: Optional[str] = None


def set_vllm_api_url(url: str):
    """Initialize the base URL for vLLM API calls."""
    global VLLM_API_URL
    VLLM_API_URL = url
    logging.info("vLLM API → %s", VLLM_API_URL)


def _api(endpoint: str, payload: Dict[str, Any], timeout: int = 300):
    assert VLLM_API_URL, "VLLM API URL not set"
    url = f"{VLLM_API_URL}/v1/{endpoint}"
    for attempt in range(3):
        try:
            logging.debug("POST %s try %d payload %s", endpoint, attempt + 1, payload)
            r = API.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                if r.headers.get("Content-Type", "").startswith("application/json"):
                    return r.json()
                return r.text or True
            r.raise_for_status()
        except Exception as e:
            logging.warning("API error %s - attempt %d/3", e, attempt + 1)
            time.sleep(2 * (attempt + 1))
    logging.error("API %s failed after retries", endpoint)
    return None


def load_adapter(path: str, name: str) -> bool:
    return _api("load_lora_adapter", {"lora_name": name, "lora_path": path}) is not None


def unload_adapter(name: str) -> bool:
    _api("unload_lora_adapter", {"lora_name": name}); return True


def generate(
    prompts: List[str], model: str, sampling: Dict[str, Any], stop_ids: List[int]
) -> Optional[List[Dict[str, Any]]]:
    payload = {"model": model, "prompt": prompts, **sampling, "stop_token_ids": stop_ids}
    res = _api("completions", payload, timeout=120*len(prompts))
    return res.get("choices") if isinstance(res, dict) else None


# -------------------  SQUAD HELPERS  ---------------------------------- #
def format_answer_prompts(
    q_batch: List[Dict[str, str]],
    instruct_model: bool,
    chain_of_thought: bool = False,
    thinking_mode: bool = False,
) -> List[str]:
    if chain_of_thought:
        SQUAD_ANSWER_TEMPLATE = SQUAD_ANSWER_TEMPLATE_BASE_COT
    else:
        SQUAD_ANSWER_TEMPLATE = (
            SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT if instruct_model else SQUAD_ANSWER_TEMPLATE_BASE
        )
    prompts = [SQUAD_ANSWER_TEMPLATE.format(question=q["question"]) for q in q_batch]
    # Instruct + non-thinking: force empty think block (Qwen3 enable_thinking=false).
    if instruct_model and not thinking_mode and not chain_of_thought:
        prompts = [p + QWEN3_DISABLE_THINKING_PREFIX for p in prompts]
    return prompts


def format_grade_prompts(
    q_batch: List[Dict[str, str]], preds: List[str]
) -> List[str]:
    return [
        SQUAD_GRADE_TEMPLATE.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip(),
        )
        for q, p in zip(q_batch, preds)
    ]

_yes_re = re.compile(r"\b(yes)\b", re.I)
_no_re  = re.compile(r"\b(no)\b",  re.I)

_gpt4: OpenAI | None = None

_final_ans_re = re.compile(
    r"(?:^|\n)\s*final\s*answer\s*[:\-]\s*(.*)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

def extract_final_answer(text: str) -> str:
    """
    Return only the content after a 'Final answer:' (case-insensitive) marker.
    If no marker is present, return 'idk' (so it will be graded as incorrect).
    """
    if not text:
        return "idk"
    m = _final_ans_re.search(text.strip())
    return (m.group(1).strip() if m else "idk").strip()

def _client() -> OpenAI:
    """Return a singleton OpenAI client (reads OPENAI_API_KEY from env)."""
    global _gpt4
    if _gpt4 is None:
        _gpt4 = OpenAI(api_key="sk-zTULhCULj7GF1HL30fjKc41ieclje27zB4qaWyV5BeQF534K", base_url="https://quanzil.com/v1")
    return _gpt4

def grade_with_gpt4(prompts: List[str]) -> List[bool]:
    """
    Take already-formatted grading prompts, send each to GPT-4.1,
    and return the yes/no verdicts as booleans.
    """
    verdicts: List[bool] = []
    client: OpenAI = _client()

    for p in prompts:
        for attempt in range(3):
            try:
                r = client.responses.create(model="gpt-4.1", input=p)
                verdicts.append(parse_yes_no(r.output_text))
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            try:
                chat = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": p}],
                )
                verdicts.append(parse_yes_no(chat.choices[0].message.content))
            except Exception:
                verdicts.append(False)  # couldn't grade this prompt

    return verdicts


def parse_yes_no(text: str) -> bool:
    """Return True for yes, False for no or ambiguous responses."""
    if _yes_re.search(text) and not _no_re.search(text):
        return True
    return False

def _split_segments(text: str) -> List[str]:
    return [seg.strip() for seg in text.split("---") if seg.strip()]

MAX_TRAIN_SEQS_PER_COMPLETION=30

def build_train_sequences(
    completion_raw: str,
    context: str,
    title: str,
    *,
    split_newlines: bool = False,
    add_context: bool = True,
) -> List[str]:
    """
    Turn a raw completion + article context into the list of sequences
    that the inner-loop fine-tuning consumes.

    - `---` splits into separate training examples  
    - if `split_newlines`, each line inside a segment becomes its own example  
    - the original article context is optionally appended as the last example
      (disable via --no_add_context / add_context=False)
    - each example is prefixed with the title
    - if the second sequence begins with "1.", remove the first one
    """
    # For chain-of-thought, keep only content after the first "\nImplications:" marker
    m = re.search(r"\nImplications:\s*", completion_raw)
    if m:
        completion_raw = completion_raw[m.end():].lstrip()

    segs = _split_segments(completion_raw) or [completion_raw.strip()]
    if split_newlines:
        if re.search(r'Question\s+\d+:', completion_raw) and re.search(r'Answer\s*:', completion_raw): 
            # deal with self-QA responses
            # split wherever a new "Question N:" begins
            segs = re.split(r'\n(?=Question\s+\d+:)', completion_raw.strip())
            # ensure the very first segment has an explicit "Question 1:" prefix
            if not segs[0].lstrip().startswith("Question"):
                segs[0] = "Question 1: " + segs[0].strip()
        else: 
            # deal with responses where first line is along the lines of "Sure, let's give a list of implications:"
            segs = [ln.strip() for seg in segs for ln in seg.splitlines() if ln.strip()]
            if len(segs) > 1 and segs[1].startswith("1."):
                segs = segs[1:]

    if len(segs) > MAX_TRAIN_SEQS_PER_COMPLETION:
        segs = segs[:MAX_TRAIN_SEQS_PER_COMPLETION]
    segs = [s for s in segs if s.strip()]
    seqs = [TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=s) for s in segs]
    if add_context:
        seqs.append(TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=context.strip()))
    return seqs

# -------------------  PROXY GRADING  ---------------------------------- #
# This is a proxy used to evaluate the quality of synthetic data generated by the model.
# It is meant as an alternative to the full meta-learning inner loop, and can be applied
# in TTT_server.py by setting --reward_mode to "proxy" or "both". This rubric is not heavily tuned
# and likely could be improved further.
PROXY_SCORE_TEMPLATE = (
    "You are to evaluate a list of implications that are derived (directly or indirectly) from the provided document. Be critical and fair.\n"
    "For each of the four criteria below, rate on a 1-5 integer scale (higher is better). We want to reward implication lists that are longer (relative to the size of the original document), more diverse (statements should be less repetitive), higher in quality, and more correct (supported by the document).\n"
    "For each criterion, think briefly and then output exactly this format:\n"
    "Length: <1-5> - <one sentence rationale>\n"
    "Diversity: <1-5> - <one sentence rationale>\n"
    "Quality: <1-5> - <one sentence rationale>\n"
    "Correctness: <1-5> - <one sentence rationale>\n"
    "After that, output the sum of the 4 scores as: Final Score: <integer>\n\n"
    "Title: {title}\n"
    "Document:\n{context}\n\n"
    "Implications:\n{completion}\n"
)

_proxy_len_re = re.compile(r"^\s*Length:\s*(\d+)", re.I | re.M)
_proxy_div_re = re.compile(r"^\s*Diversity:\s*(\d+)", re.I | re.M)
_proxy_quality_re = re.compile(r"^\s*Quality:\s*(\d+)", re.I | re.M)
_proxy_correct_re = re.compile(r"^\s*Correct(ness)?:\s*(\d+)", re.I | re.M)
_proxy_final_re = re.compile(r"^\s*Final\s*Score\s*:\s*(\d+)", re.I | re.M)

def build_proxy_prompt(title: str, context: str, completion: str) -> str:
    return PROXY_SCORE_TEMPLATE.format(title=title, context=context, completion=completion)

def _to_int_clamped(x: str, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return 1

def parse_proxy_scores(text: str) -> Dict[str, int]:
    """Extract integer sub-scores and final score from GPT output."""
    length = _to_int_clamped((_proxy_len_re.search(text) or [None, "1"])[1])
    diversity = _to_int_clamped((_proxy_div_re.search(text) or [None, "1"])[1])
    quality = _to_int_clamped((_proxy_quality_re.search(text) or [None, "1"])[1])
    correctness = _to_int_clamped(((_proxy_correct_re.search(text) or [None, None, "1"])[2]))
    final_match = _proxy_final_re.search(text)
    if final_match:
        final_score = int(final_match.group(1))
    else:
        final_score = length + diversity + quality + correctness
    # clamp final into [4, 20]
    final_score = max(4, min(20, final_score))
    return {
        "length": length,
        "diversity": diversity,
        "quality": quality,
        "correctness": correctness,
        "final": final_score,
    }

def score_proxy_with_gpt4(title: str, context: str, completion: str) -> Dict[str, int]:
    """
    Ask GPT-4.1 to score the completion on four criteria (1-5) and a Final Score.
    Returns a dict with keys: length, diversity, quality, correctness, final.
    """
    client: OpenAI = _client()
    prompt = build_proxy_prompt(title, context, completion)
    text_out = ""
    for attempt in range(3):
        try:
            r = client.responses.create(model="gpt-4.1", input=prompt)
            text_out = r.output_text or ""
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if not text_out:
        return {"length": 1, "diversity": 1, "quality": 1, "correctness": 1, "final": 4}
    return parse_proxy_scores(text_out)


# -------------------  TASK SIMILARITY (Baseline 3)  -------------------- #
TASK_SIMILARITY_PROMPT_VERSION = "task_sim_v1"

TASK_SIMILARITY_TEMPLATE = (
    "You are evaluating whether continual-learning tasks could share the same LoRA adapter subspace.\n"
    "For each historical candidate, score how similar its implications are to the current task's "
    "implications in terms of knowledge theme, reasoning patterns, and whether the same LoRA "
    "fine-tuning subspace could serve both tasks.\n\n"
    "Score each candidate similarity s in [0, 1] where:\n"
    "- s=1.0 means highly similar (could share LoRA subspace)\n"
    "- s=0.0 means completely dissimilar (need orthogonal subspace)\n\n"
    "Do NOT output lambda or orthogonality weights. Only output similarity scores.\n\n"
    "Current passage:\n"
    "Title: {title}\n"
    "Context:\n{context}\n\n"
    "Current implications (I_t):\n{current_implications}\n\n"
    "Historical candidates:\n{candidates_block}\n\n"
    "Respond with JSON only:\n"
    '{{\n'
    '  "scores": [\n'
    '    {{"task_id": "<id>", "similarity": 0.72}},\n'
    "    ...\n"
    "  ],\n"
    '  "most_similar_task_id": "<id>",\n'
    '  "most_similar_score": 0.72\n'
    "}}\n"
)

_sim_json_re = re.compile(r"\{[\s\S]*\}")
_sim_score_line_re = re.compile(
    r"Similarity\s*:\s*([\d.]+)", re.I
)


def _format_similarity_candidates(
    candidates: List[Dict[str, str]],
) -> str:
    blocks: List[str] = []
    for i, c in enumerate(candidates, 1):
        blocks.append(
            f"Candidate {i} (task_id={c['task_id']}):\n{c['I_t_text']}\n"
        )
    return "\n".join(blocks)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_task_similarity_response(
    text: str,
    candidate_ids: List[str],
) -> Dict[str, Any]:
    """Parse GPT-4.1 task similarity JSON or fallback patterns."""
    text = (text or "").strip()
    result: Dict[str, Any] = {
        "scores": [],
        "most_similar_task_id": None,
        "most_similar_score": 0.0,
        "raw": text,
    }
    if not text:
        return result

    json_match = _sim_json_re.search(text)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            scores = payload.get("scores") or []
            parsed_scores: List[Dict[str, Any]] = []
            for entry in scores:
                tid = entry.get("task_id")
                sim = _clamp01(float(entry.get("similarity", 0.0)))
                if tid:
                    parsed_scores.append({"task_id": tid, "similarity": sim})
            result["scores"] = parsed_scores
            if payload.get("most_similar_task_id"):
                result["most_similar_task_id"] = payload["most_similar_task_id"]
            if payload.get("most_similar_score") is not None:
                result["most_similar_score"] = _clamp01(
                    float(payload["most_similar_score"])
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not result["scores"] and candidate_ids:
        line_match = _sim_score_line_re.search(text)
        if line_match:
            sim = _clamp01(float(line_match.group(1)))
            result["most_similar_task_id"] = candidate_ids[0]
            result["most_similar_score"] = sim
            result["scores"] = [
                {"task_id": candidate_ids[0], "similarity": sim}
            ]

    if result["scores"] and result["most_similar_task_id"] is None:
        best = max(result["scores"], key=lambda x: x["similarity"])
        result["most_similar_task_id"] = best["task_id"]
        result["most_similar_score"] = best["similarity"]
    elif result["most_similar_task_id"] and not result["scores"]:
        result["scores"] = [
            {
                "task_id": result["most_similar_task_id"],
                "similarity": result["most_similar_score"],
            }
        ]

    if (
        result["most_similar_task_id"]
        and result["most_similar_score"] == 0.0
        and result["scores"]
    ):
        best = max(result["scores"], key=lambda x: x["similarity"])
        result["most_similar_task_id"] = best["task_id"]
        result["most_similar_score"] = best["similarity"]

    return result


def score_task_similarity_with_gpt4(
    *,
    title: str,
    context: str,
    current_implications: str,
    candidates: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Ask GPT-4.1 to score task similarity for Top-K historical candidates.

    Returns dict with keys: most_similar_task_id, most_similar_score, scores, raw.
    On failure, returns conservative fallback (score=0.0 → lambda=1.0).
    """
    if not candidates:
        return {
            "most_similar_task_id": None,
            "most_similar_score": 0.0,
            "scores": [],
            "raw": "",
            "prompt": "",
            "fallback": True,
        }

    candidate_ids = [c["task_id"] for c in candidates]
    prompt = TASK_SIMILARITY_TEMPLATE.format(
        title=title,
        context=context,
        current_implications=current_implications,
        candidates_block=_format_similarity_candidates(candidates),
    )
    client = _client()
    text_out = ""
    for attempt in range(3):
        try:
            r = client.responses.create(model="gpt-4.1", input=prompt)
            text_out = r.output_text or ""
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if not text_out:
        for attempt in range(3):
            try:
                chat = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                )
                text_out = chat.choices[0].message.content or ""
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))

    if not text_out:
        return {
            "most_similar_task_id": None,
            "most_similar_score": 0.0,
            "scores": [],
            "raw": "",
            "prompt": prompt,
            "fallback": True,
        }

    parsed = parse_task_similarity_response(text_out, candidate_ids)
    parsed["prompt"] = prompt
    parsed["fallback"] = False
    if parsed["most_similar_task_id"] not in candidate_ids:
        if parsed["scores"]:
            best = max(parsed["scores"], key=lambda x: x["similarity"])
            parsed["most_similar_task_id"] = best["task_id"]
            parsed["most_similar_score"] = best["similarity"]
        else:
            parsed["most_similar_task_id"] = candidate_ids[0]
            parsed["most_similar_score"] = 0.0
            parsed["fallback"] = True
    return parsed
