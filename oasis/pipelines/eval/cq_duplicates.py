from __future__ import annotations

from collections import Counter
from typing import List


def shingles(tokens: List[str], k: int = 3) -> set[tuple[str, ...]]:
    if k <= 0:
        return set()
    return set(tuple(tokens[i : i + k]) for i in range(max(0, len(tokens) - k + 1)))


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_duplicate_stats(texts: List[str]) -> tuple[list[dict], int, float]:
    counts = Counter(texts)
    duplicates = [
        {"normalized_text": text, "count": count}
        for text, count in counts.items()
        if count > 1
    ]
    duplicates_sorted = sorted(duplicates, key=lambda item: item["count"], reverse=True)
    exact_duplicates_count = sum(item["count"] - 1 for item in duplicates_sorted)
    unique_ratio = (len(counts) / len(texts)) if texts else 0.0
    return duplicates_sorted, exact_duplicates_count, unique_ratio


def near_duplicate_pairs(
    token_lists: List[List[str]],
    ids: List[str],
    normalized_texts: List[str],
    *,
    threshold: float,
    shingle_size: int,
    progress=None,
) -> tuple[list[dict], int]:
    shingle_sets = [shingles(tokens, k=shingle_size) for tokens in token_lists]
    pairs = []
    total_pairs = 0
    iterator = range(len(shingle_sets))
    if progress is not None:
        iterator = progress(iterator, desc="Jaccard pairs")
    for i in iterator:
        for j in range(i + 1, len(shingle_sets)):
            total_pairs += 1
            score = jaccard(shingle_sets[i], shingle_sets[j])
            if score >= threshold:
                pairs.append(
                    {
                        "id_a": ids[i],
                        "id_b": ids[j],
                        "score": score,
                        "text_a": normalized_texts[i],
                        "text_b": normalized_texts[j],
                    }
                )
    return pairs, total_pairs
