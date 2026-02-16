# Подготовка датасетов к BigARTM: VW + словарь. Логика из 01_prepare_vw и 02_build_dictionary.

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import artm

# Токенизация и VW (из topic_modeling_bigartm/utils и 01)
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
    "we", "our", "you", "your", "from", "can", "may", "might", "should", "would",
    "also", "than", "which", "while", "where", "when", "who", "whom", "what",
    "how", "do", "does", "did", "done", "using", "use", "used", "via", "based",
    "results", "result", "table", "figure", "fig", "section", "introduction",
    "related", "work", "conclusion", "references", "paper", "study", "studies",
    "analysis", "method", "methods", "approach", "approaches", "data", "dataset",
    "datasets", "model", "models", "task", "tasks", "problem", "problems",
}
BIBLIO_STOP = {
    "arxiv", "ieee", "acm", "proceedings", "conference", "workshop", "symposium",
    "journal", "transactions", "doi", "isbn", "issn", "pp", "vol", "volume", "issue", "pages", "page",
    "cvpr", "iccv", "eccv", "neurips", "nips", "icml", "iclr", "aaai", "ijcai", "acl", "emnlp", "naacl", "coling",
    "kdd", "icdm",
}
FRAGMENT_SUFFIXES = {"ure", "ble", "nal", "ver"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TOKEN_RE = re.compile(r"[a-z][a-z\-]{1,}")
PUNCT_STRIP = ".,:;!?()[]{}<>\"'`"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fix_hyphenation(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[‐\-‒–—−]", "-", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text


def _clean_text(text: str) -> str:
    text = _fix_hyphenation(text)
    text = text.replace("\x00", " ")
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return text


def _tokenize_with_stats(text: str, stopwords: set[str]) -> tuple[list[str], dict]:
    text = _clean_text(text.lower())
    raw_tokens = TOKEN_RE.findall(text)
    stats = {"raw_tokens": len(raw_tokens), "stopword_removed": 0, "short_removed": 0}
    tokens = []
    for tok in raw_tokens:
        tok = tok.strip(PUNCT_STRIP).strip("-")
        if not tok:
            stats["short_removed"] += 1
            continue
        if tok in stopwords:
            stats["stopword_removed"] += 1
            continue
        if len(tok) < 2:
            stats["short_removed"] += 1
            continue
        tokens.append(tok)
    return tokens, stats


def _sanitize_vw_token(token: str) -> str:
    out = []
    for ch in token:
        if ch.isspace():
            out.append("_")
        elif ch in (":", "|"):
            out.append("_")
        elif 32 <= ord(ch) <= 126:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _generate_ngrams(tokens: list[str], n: int, separator: str = "_") -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [separator.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _load_drop_tokens_file(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip().lower() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.strip().startswith("#")}


def _make_doc_id(path: Path) -> str:
    safe = path.stem.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe if safe else "doc"


def prepare_vw(
    input_dir: Path,
    output_dir: Path,
    modalities: set[str],
    min_doc_length: int = 20,
    drop_min_token_len: int = 3,
    keep_short_tokens: str = "ai,ml,dl,cv,rl,nn",
    drop_tokens_file: str | Path | None = None,
    drop_biblio: bool = True,
) -> int:
    """Генерирует vw.txt и preprocess/* в output_dir. Возвращает 0 при успехе."""
    valid = {"uni", "bi", "tri"}
    if not modalities.issubset(valid):
        return 1
    generate_uni = "uni" in modalities
    generate_bi = "bi" in modalities
    generate_tri = "tri" in modalities
    _ensure_dir(output_dir)
    preprocess_dir = output_dir / "preprocess"
    _ensure_dir(preprocess_dir)
    vw_path = output_dir / "vw.txt"
    tokenized_path = preprocess_dir / "tokenized.jsonl"
    doc_map_path = preprocess_dir / "doc_map.json"
    vocab_stats_path = preprocess_dir / "vocab_stats.json"
    for p in (vw_path, tokenized_path, doc_map_path, vocab_stats_path):
        if p.exists():
            p.unlink()
    keep_short = {t.strip().lower() for t in keep_short_tokens.split(",") if t.strip()}
    custom_drop = _load_drop_tokens_file(drop_tokens_file)
    doc_map: dict[str, str] = {}
    token_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    total_tokens = 0
    kept_docs = 0
    dropped_docs = 0
    total_docs = 0
    doc_lengths: list[int] = []
    file_list = sorted(input_dir.glob("*.txt"))
    total_docs = len(file_list)
    with vw_path.open("w", encoding="utf-8") as vw_f, tokenized_path.open("w", encoding="utf-8") as tok_f:
        for path in file_list:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tokens_raw, _ = _tokenize_with_stats(text, DEFAULT_STOPWORDS)
            tokens = []
            for tok in tokens_raw:
                low = tok.lower()
                if drop_biblio and low in BIBLIO_STOP:
                    continue
                if low in custom_drop:
                    continue
                if len(low) < drop_min_token_len and low not in keep_short:
                    continue
                if len(low) <= 2 and low.isalpha() and low not in keep_short:
                    continue
                s = _sanitize_vw_token(tok)
                if s:
                    tokens.append(s)
            if len(tokens) < min_doc_length:
                dropped_docs += 1
                continue
            doc_id = _make_doc_id(path)
            doc_map[doc_id] = path.name
            kept_docs += 1
            total_tokens += len(tokens)
            token_counts.update(tokens)
            doc_lengths.append(len(tokens))
            bigrams = []
            trigrams = []
            if generate_bi:
                bigrams = [_sanitize_vw_token(bg) for bg in _generate_ngrams(tokens, 2) if _sanitize_vw_token(bg)]
                bigram_counts.update(bigrams)
            if generate_tri:
                trigrams = [_sanitize_vw_token(tg) for tg in _generate_ngrams(tokens, 3) if _sanitize_vw_token(tg)]
                trigram_counts.update(trigrams)
            vw_parts = [doc_id]
            if generate_uni:
                vw_parts.append("|text " + " ".join(tokens))
            if generate_bi and bigrams:
                vw_parts.append("|bigrams " + " ".join(bigrams))
            if generate_tri and trigrams:
                vw_parts.append("|trigrams " + " ".join(trigrams))
            vw_f.write(" ".join(vw_parts) + "\n")
            doc_data = {"doc_id": doc_id, "tokens": tokens}
            if generate_bi:
                doc_data["bigrams"] = bigrams
            if generate_tri:
                doc_data["trigrams"] = trigrams
            tok_f.write(json.dumps(doc_data) + "\n")
    modality_names = []
    if generate_uni:
        modality_names.append("text")
    if generate_bi:
        modality_names.append("bigrams")
    if generate_tri:
        modality_names.append("trigrams")
    vocab_counts = dict(token_counts.most_common(50)) if token_counts else {}
    unique_tokens = len(token_counts)
    hapax = sum(1 for _, c in token_counts.items() if c == 1)
    doc_len_stats = {"min": min(doc_lengths) if doc_lengths else 0, "median": 0, "p95": 0, "p99": 0}
    if doc_lengths:
        srt = sorted(doc_lengths)
        doc_len_stats["median"] = srt[len(srt) // 2]
        doc_len_stats["p95"] = srt[int(0.95 * (len(srt) - 1))] if len(srt) > 1 else srt[0]
        doc_len_stats["p99"] = srt[int(0.99 * (len(srt) - 1))] if len(srt) > 1 else srt[0]
    doc_map_path.write_text(json.dumps(doc_map, indent=2), encoding="utf-8")
    vocab_stats_path.write_text(json.dumps({
        "documents": kept_docs,
        "documents_total": total_docs,
        "documents_dropped": dropped_docs,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "hapax_share": (hapax / unique_tokens) if unique_tokens else 0.0,
        "doc_length": doc_len_stats,
        "top_tokens": vocab_counts,
        "modalities": modality_names,
        "unique_bigrams": len(bigram_counts) if generate_bi else 0,
        "unique_trigrams": len(trigram_counts) if generate_tri else 0,
    }, indent=2), encoding="utf-8")
    return 0


def _dict_size(dct) -> int | None:
    for attr in ("num_entries", "num_entries_"):
        if hasattr(dct, attr):
            return int(getattr(dct, attr))
    try:
        return len(dct)
    except TypeError:
        return None


def _dict_size_fallback(dct, tmp_dir: Path) -> int | None:
    try:
        tmp_path = tmp_dir / "dictionary.txt"
        dct.save_text(str(tmp_path))
        count = sum(1 for line in tmp_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.strip().lower().startswith("token"))
        tmp_path.unlink(missing_ok=True)
        return count
    except Exception:
        return None


def _filter_dictionary_text(
    src: Path,
    dst: Path,
    min_len: int,
    keep_short: set[str],
    drop_tokens: set[str],
    drop_biblio: bool,
) -> dict:
    removed: Counter = Counter()
    kept: list[str] = []
    header_kept = False
    short_alpha_re = re.compile(r"^[a-z]{1,2}$")
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if parts[0].lower() == "token" and len(parts) >= 5 and not header_kept:
                kept.append(line)
                header_kept = True
                continue
            if len(parts) < 5:
                kept.append(line)
                removed["malformed_kept"] += 1
                continue
            token = parts[0].lower()
            if token.endswith("-"):
                removed["trailing_hyphen"] += 1
                continue
            if drop_biblio and token in BIBLIO_STOP:
                removed["biblio_stoplist"] += 1
                continue
            if token in drop_tokens:
                removed["custom_stoplist"] += 1
                continue
            if len(token) == 3 and token in FRAGMENT_SUFFIXES:
                removed["fragment_suffix"] += 1
                continue
            if len(token) < min_len and token not in keep_short:
                removed["too_short"] += 1
                continue
            if short_alpha_re.match(token) and token not in keep_short:
                removed["short_alpha"] += 1
                continue
            kept.append(line)
    dst.write_text("".join(kept), encoding="utf-8")
    return dict(removed)


def build_dictionary(
    output_dir: Path,
    vw_path: Path | None = None,
    min_df: int = 2,
    max_df_rate: float = 0.7,
    min_tf: int = 2,
    max_dictionary_size: int = 50000,
    drop_min_token_len: int = 3,
    keep_short_tokens: str = "ai,ml,dl,cv,rl,nn",
    drop_tokens_file: str | Path | None = None,
    drop_biblio: bool = True,
) -> int:
    """Строит словарь и батчи в output_dir. vw_path по умолчанию output_dir/vw.txt. Возвращает 0 при успехе."""
    vw = vw_path or (output_dir / "vw.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = output_dir / "batches"
    if batches_dir.exists():
        for item in batches_dir.iterdir():
            if item.is_dir():
                for sub in item.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                item.rmdir()
            else:
                item.unlink()
    batches_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir = output_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    batch_vectorizer = artm.BatchVectorizer(
        data_path=str(vw),
        data_format="vowpal_wabbit",
        target_folder=str(batches_dir),
    )
    dictionary = batch_vectorizer.dictionary
    before_size = _dict_size(dictionary)
    if before_size is None:
        before_size = _dict_size_fallback(dictionary, output_dir)
    dictionary.filter(min_df=min_df, max_df_rate=max_df_rate, min_tf=min_tf, max_dictionary_size=max_dictionary_size)
    after_size = _dict_size(dictionary)
    if after_size is None:
        after_size = _dict_size_fallback(dictionary, output_dir)
    dict_path = output_dir / "dictionary.dict"
    dict_text_path = output_dir / "dictionary.txt"
    dict_filtered_path = output_dir / "dictionary.filtered.txt"
    if dict_path.exists():
        dict_path.unlink()
    dictionary.save(str(dict_path))
    dictionary.save_text(str(dict_text_path))
    keep_short = {t.strip().lower() for t in keep_short_tokens.split(",") if t.strip()}
    custom_drop = _load_drop_tokens_file(drop_tokens_file)
    _filter_dictionary_text(dict_text_path, dict_filtered_path, min_len=drop_min_token_len, keep_short=keep_short, drop_tokens=custom_drop, drop_biblio=drop_biblio)
    dictionary.load_text(str(dict_filtered_path))
    if dict_path.exists():
        dict_path.unlink()
    if dict_text_path.exists():
        dict_text_path.unlink()
    dictionary.save(str(dict_path))
    dictionary.save_text(str(dict_text_path))
    post_size = _dict_size(dictionary) or _dict_size_fallback(dictionary, output_dir)
    tokens_sample: list[str] = []
    try:
        for line in dict_text_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 3 or parts[0].lower() in ("token", "name"):
                continue
            tokens_sample.append(parts[0])
            if len(tokens_sample) >= 50:
                break
    except FileNotFoundError:
        pass
    manifest_path = preprocess_dir / "dict_build_run.json"
    manifest_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vw_path": str(vw),
        "batches_dir": str(batches_dir),
        "dictionary_path": str(dict_path),
        "params": {"min_df": min_df, "max_df_rate": max_df_rate, "min_tf": min_tf, "max_dictionary_size": max_dictionary_size},
        "stats": {"before_size": before_size, "after_size": after_size, "after_size_postfilter": post_size, "tokens_sample": tokens_sample},
    }, indent=2), encoding="utf-8")
    return 0


def run_all(
    root_dir: Path,
    data_root: Path,
    dataset_names: list[str],
    artifacts_root: Path | None = None,
    modalities: set[str] | None = None,
) -> int:
    """По каждому датасету выполняет prepare_vw и build_dictionary. Без вызова внешних скриптов."""
    art_root = artifacts_root or (root_dir / "artifacts")
    mods = modalities or {"uni", "bi"}
    failed = 0
    for dataset in dataset_names:
        input_dir = data_root / dataset / "topics"
        if not input_dir.exists() or not list(input_dir.glob("*.txt")):
            continue
        output_dir = art_root / dataset / "artm"
        if prepare_vw(input_dir, output_dir, mods) != 0:
            failed += 1
            continue
        if build_dictionary(output_dir) != 0:
            failed += 1
    return 1 if failed else 0
