# Оценка числа тем K по корпусу: coherence, diversity, stability по bootstrap-запускам BigARTM.

from __future__ import annotations

import json
import math
import random
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import artm

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_RE = re.compile(r"`[^`]+`")
FENCE_RE = re.compile(r"```[\s\S]*?```")


@dataclass
class CoocStats:
    doc_count: int
    df: Counter
    pair_df: Counter


def clean_markdown(text: str) -> str:
    text = FENCE_RE.sub(" ", text)
    text = CODE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    return text


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    cleaned = clean_markdown(text.lower())
    return [token for token in TOKEN_RE.findall(cleaned) if token not in stopwords]


def load_corpus(input_dir: Path) -> tuple[list[list[str]], list[str]]:
    files = sorted(input_dir.glob("*.txt"))
    tokenized_docs: list[list[str]] = []
    doc_ids: list[str] = []
    for path in files:
        tokens = tokenize(path.read_text(encoding="utf-8", errors="ignore"), stopwords=set())
        if tokens:
            tokenized_docs.append(tokens)
            doc_ids.append(path.stem)
    if not tokenized_docs:
        raise ValueError("В директории нет документов: %s" % input_dir)
    return tokenized_docs, doc_ids


def build_cooc_stats(tokenized_docs: list[list[str]]) -> CoocStats:
    df: Counter = Counter()
    pair_df: Counter = Counter()
    for tokens in tokenized_docs:
        uniq = list(dict.fromkeys(tokens))
        uniq_set = set(uniq)
        for token in uniq_set:
            df[token] += 1
        for i, wi in enumerate(uniq):
            for wj in uniq[i + 1 :]:
                if wi == wj:
                    continue
                a, b = sorted((wi, wj))
                pair_df[(a, b)] += 1
    return CoocStats(doc_count=len(tokenized_docs), df=df, pair_df=pair_df)


def npmi(wi: str, wj: str, stats: CoocStats) -> float:
    if wi == wj:
        return 0.0
    a, b = sorted((wi, wj))
    p_i = stats.df[wi] / stats.doc_count
    p_j = stats.df[wj] / stats.doc_count
    p_ij = stats.pair_df[(a, b)] / stats.doc_count
    if p_i <= 0 or p_j <= 0 or p_ij <= 0:
        return 0.0
    pmi = math.log(p_ij / (p_i * p_j))
    return pmi / (-math.log(p_ij))


def coherence_npmi(topics: list[list[str]], stats: CoocStats, top_m: int) -> float:
    topic_scores: list[float] = []
    for topic in topics:
        words = topic[:top_m]
        if len(words) < 2:
            continue
        pair_scores: list[float] = []
        for i, wi in enumerate(words):
            for wj in words[i + 1 :]:
                pair_scores.append(npmi(wi, wj, stats))
        if pair_scores:
            topic_scores.append(sum(pair_scores) / len(pair_scores))
    if not topic_scores:
        return 0.0
    return sum(topic_scores) / len(topic_scores)


def topic_diversity(topics: list[list[str]], top_m: int) -> float:
    words: list[str] = []
    for topic in topics:
        words.extend(topic[:top_m])
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def topic_set_similarity(a: list[list[str]], b: list[list[str]], top_m: int) -> float:
    if not a or not b:
        return 0.0
    a_sets = [set(topic[:top_m]) for topic in a]
    b_sets = [set(topic[:top_m]) for topic in b]
    remaining = list(range(len(b_sets)))
    scores: list[float] = []
    for aset in a_sets:
        best_j = -1
        best_score = -1.0
        for j in remaining:
            score = jaccard(aset, b_sets[j])
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            scores.append(best_score)
            remaining.remove(best_j)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def stability_score(runs_topics: list[list[list[str]]], top_m: int) -> float:
    if len(runs_topics) < 2:
        return 0.0
    sims: list[float] = []
    for i in range(len(runs_topics)):
        for j in range(i + 1, len(runs_topics)):
            sims.append(topic_set_similarity(runs_topics[i], runs_topics[j], top_m))
    return sum(sims) / len(sims)


def train_bigartm_topics(
    tokenized_docs: list[list[str]],
    k: int,
    passes: int,
    top_n: int,
    seed: int,
) -> list[list[str]]:
    with tempfile.TemporaryDirectory(prefix="kselect_") as tmp:
        tmp_dir = Path(tmp)
        vw_path = tmp_dir / "corpus.vw"
        with vw_path.open("w", encoding="utf-8") as f:
            for idx, tokens in enumerate(tokenized_docs):
                counts = Counter(tokens)
                bow = " ".join("%s:%d" % (t, c) for t, c in counts.items())
                f.write("doc_%d |text %s\n" % (idx, bow))
        batches_dir = tmp_dir / "batches"
        vectorizer = artm.BatchVectorizer(
            data_path=str(vw_path),
            data_format="vowpal_wabbit",
            target_folder=str(batches_dir),
        )
        model = artm.ARTM(
            num_topics=k,
            dictionary=vectorizer.dictionary,
            class_ids={"text": 1.0},
            seed=seed,
        )
        model.scores.add(artm.TopTokensScore(name="top_tokens", class_id="text", num_tokens=top_n))
        model.fit_offline(batch_vectorizer=vectorizer, num_collection_passes=passes)
        last_tokens = model.score_tracker["top_tokens"].last_tokens
        topics: list[list[str]] = []
        for topic_name in sorted(last_tokens.keys()):
            words = [w for w in last_tokens[topic_name] if isinstance(w, str) and w]
            if words:
                topics.append(words[:top_n])
        return topics


def minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def choose_k(
    k_values: list[int],
    coherence_values: list[float],
    diversity_values: list[float],
    stability_values: list[float],
    k_penalty_lambda: float,
) -> dict:
    coh_n = minmax_norm(coherence_values)
    div_n = minmax_norm(diversity_values)
    stab_n = minmax_norm(stability_values)
    k_n = minmax_norm([float(k) for k in k_values])
    scores = [
        0.5 * coh_n[i] + 0.25 * div_n[i] + 0.25 * stab_n[i] - k_penalty_lambda * k_n[i]
        for i in range(len(k_values))
    ]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return {
        "k_values": k_values,
        "coherence": coherence_values,
        "diversity": diversity_values,
        "stability": stability_values,
        "score": scores,
        "k_best": k_values[best_idx],
    }


def bootstrap_docs(tokenized_docs: list[list[str]], frac: float, seed: int) -> list[list[str]]:
    random.seed(seed)
    n_total = len(tokenized_docs)
    n_sample = max(2, int(round(n_total * frac)))
    indices = [random.randrange(n_total) for _ in range(n_sample)]
    return [tokenized_docs[idx] for idx in indices]


def process_dataset(
    input_dir: Path,
    k_min: int,
    k_max: int,
    k_step: int,
    runs: int,
    bootstrap_frac: float,
    top_n: int,
    npmi_top_m: int,
    stability_top_m: int,
    bigartm_passes: int,
    seed: int,
    k_penalty_lambda: float,
) -> dict:
    tokenized_docs, _ = load_corpus(input_dir)
    stats = build_cooc_stats(tokenized_docs)
    k_values = list(range(k_min, k_max + 1, k_step))
    coherence_values: list[float] = []
    diversity_values: list[float] = []
    stability_values: list[float] = []
    for k in k_values:
        run_topics: list[list[list[str]]] = []
        run_coh: list[float] = []
        run_div: list[float] = []
        for run_idx in range(runs):
            boot = bootstrap_docs(tokenized_docs, frac=bootstrap_frac, seed=seed + run_idx + 997 * k)
            topics = train_bigartm_topics(boot, k=k, passes=bigartm_passes, top_n=top_n, seed=seed + run_idx)
            run_topics.append(topics)
            run_coh.append(coherence_npmi(topics, stats, top_m=npmi_top_m))
            run_div.append(topic_diversity(topics, top_m=top_n))
        coherence_values.append(sum(run_coh) / len(run_coh))
        diversity_values.append(sum(run_div) / len(run_div))
        stability_values.append(stability_score(run_topics, top_m=stability_top_m))
    result = choose_k(
        k_values=k_values,
        coherence_values=coherence_values,
        diversity_values=diversity_values,
        stability_values=stability_values,
        k_penalty_lambda=k_penalty_lambda,
    )
    result["input_dir"] = str(input_dir)
    return result


def run_all(
    data_root: Path,
    out_path: Path,
    dataset_names: list[str],
    k_min: int = 10,
    k_max: int = 100,
    k_step: int = 10,
    runs: int = 7,
    bootstrap_frac: float = 0.9,
    top_n: int = 20,
    npmi_top_m: int = 20,
    stability_top_m: int = 20,
    bigartm_passes: int = 15,
    seed: int = 42,
    k_penalty_lambda: float = 0.10,
) -> int:
    """Оценка K по всем датасетам из списка. Результат пишется в out_path."""
    targets = [
        data_root / name / "topics"
        for name in dataset_names
        if (data_root / name / "topics").exists()
    ]
    results = [
        process_dataset(
            input_dir=t,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            runs=runs,
            bootstrap_frac=bootstrap_frac,
            top_n=top_n,
            npmi_top_m=npmi_top_m,
            stability_top_m=stability_top_m,
            bigartm_passes=bigartm_passes,
            seed=seed,
            k_penalty_lambda=k_penalty_lambda,
        )
        for t in targets
    ]
    payload: dict = {"mode": "all", "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0
