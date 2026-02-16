# Обучение модели BigARTM. Логика из scripts/topic_modeling_bigartm/03_train_artm.py.

from __future__ import annotations

import json
from pathlib import Path

import artm


def parse_modality_weights(weights_str: str | None) -> dict[str, float]:
    if not weights_str:
        return {"text": 1.0}
    parts = [float(x.strip()) for x in weights_str.split(":")]
    if len(parts) == 1:
        return {"text": parts[0]}
    if len(parts) == 2:
        return {"text": parts[0], "bigrams": parts[1]}
    if len(parts) == 3:
        return {"text": parts[0], "bigrams": parts[1], "trigrams": parts[2]}
    raise ValueError("Некорректный формат весов модальностей: %s" % weights_str)


def get_available_modalities(dictionary, experiment_dir: Path | None = None) -> list[str]:
    if experiment_dir:
        vocab_path = experiment_dir / "artm" / "preprocess" / "vocab_stats.json"
        if vocab_path.exists():
            try:
                data = json.loads(vocab_path.read_text(encoding="utf-8"))
                mods = data.get("modalities", [])
                if mods:
                    return mods
            except (json.JSONDecodeError, KeyError, AttributeError):
                pass
    try:
        if hasattr(dictionary, "class_ids"):
            cids = dictionary.class_ids
            if isinstance(cids, (list, tuple, set)):
                return list(cids)
        if hasattr(dictionary, "get_class_ids"):
            return list(dictionary.get_class_ids())
    except (AttributeError, TypeError):
        pass
    return ["text"]


def build_model(
    topic_names: list[str],
    dictionary,
    theta_sparse: float,
    phi_sparse: float,
    decorrelator: float,
    seed: int,
    background_names: list[str],
    background_phi_smooth: float,
    background_theta_smooth: float,
    class_ids: dict[str, float] | None = None,
):
    if class_ids is None:
        class_ids = {"text": 1.0}
    try:
        model = artm.ARTM(
            topic_names=topic_names,
            dictionary=dictionary,
            class_ids=class_ids,
            seed=seed,
        )
        seed_applied = True
    except TypeError:
        model = artm.ARTM(
            topic_names=topic_names,
            dictionary=dictionary,
            class_ids=class_ids,
        )
        seed_applied = False
    domain_names = [t for t in topic_names if t not in background_names]
    modality_class_ids = list(class_ids.keys())

    def add_theta_sparse():
        try:
            return artm.SmoothSparseThetaRegularizer(
                tau=theta_sparse,
                name="theta_sparse",
                class_ids=modality_class_ids,
                topic_names=domain_names,
            )
        except TypeError:
            return artm.SmoothSparseThetaRegularizer(tau=theta_sparse, name="theta_sparse", topic_names=domain_names)

    def add_phi_sparse():
        try:
            return artm.SmoothSparsePhiRegularizer(
                tau=phi_sparse,
                name="phi_sparse",
                class_ids=modality_class_ids,
                topic_names=domain_names,
            )
        except TypeError:
            return artm.SmoothSparsePhiRegularizer(tau=phi_sparse, name="phi_sparse", topic_names=domain_names)

    def add_decorrelator():
        try:
            return artm.DecorrelatorPhiRegularizer(
                tau=decorrelator,
                name="decorrelator",
                class_ids=modality_class_ids,
                topic_names=domain_names,
            )
        except TypeError:
            return artm.DecorrelatorPhiRegularizer(tau=decorrelator, name="decorrelator", topic_names=domain_names)

    model.regularizers.add(add_theta_sparse())
    model.regularizers.add(add_phi_sparse())
    model.regularizers.add(add_decorrelator())
    if background_names:
        try:
            model.regularizers.add(
                artm.SmoothSparsePhiRegularizer(
                    tau=background_phi_smooth,
                    name="background_phi_smooth",
                    class_ids=modality_class_ids,
                    topic_names=background_names,
                )
            )
        except TypeError:
            model.regularizers.add(
                artm.SmoothSparsePhiRegularizer(
                    tau=background_phi_smooth,
                    name="background_phi_smooth",
                    topic_names=background_names,
                )
            )
        try:
            model.regularizers.add(
                artm.SmoothSparseThetaRegularizer(
                    tau=background_theta_smooth,
                    name="background_theta_smooth",
                    class_ids=modality_class_ids,
                    topic_names=background_names,
                )
            )
        except TypeError:
            model.regularizers.add(
                artm.SmoothSparseThetaRegularizer(
                    tau=background_theta_smooth,
                    name="background_theta_smooth",
                    topic_names=background_names,
                )
            )
    try:
        model.scores.add(artm.PerplexityScore(name="perplexity", class_ids=class_ids))
    except TypeError:
        model.scores.add(artm.PerplexityScore(name="perplexity"))
    try:
        model.scores.add(
            artm.SparsityThetaScore(
                name="sparsity_theta",
                class_id=modality_class_ids[0] if modality_class_ids else "text",
            )
        )
    except TypeError:
        model.scores.add(artm.SparsityThetaScore(name="sparsity_theta"))
    try:
        model.scores.add(
            artm.SparsityPhiScore(
                name="sparsity_phi",
                class_id=modality_class_ids[0] if modality_class_ids else "text",
            )
        )
    except TypeError:
        model.scores.add(artm.SparsityPhiScore(name="sparsity_phi"))
    for cid in modality_class_ids:
        score_name = "top_tokens_%s" % cid
        try:
            model.scores.add(artm.TopTokensScore(name=score_name, num_tokens=20, class_id=cid))
        except TypeError:
            model.scores.add(artm.TopTokensScore(name=score_name, num_tokens=20))
    if modality_class_ids:
        try:
            model.scores.add(artm.TopTokensScore(name="top_tokens", num_tokens=20, class_id=modality_class_ids[0]))
        except TypeError:
            model.scores.add(artm.TopTokensScore(name="top_tokens", num_tokens=20))
    return model, seed_applied, modality_class_ids


def _sanitize_json(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def train_artm(
    artifacts_root: Path,
    experiment: str,
    k_domain: int,
    k_background: int = 6,
    passes: int = 25,
    seed: int = 42,
    theta_sparse: float = -0.50,
    phi_sparse: float = -0.15,
    decorrelator: float = 150.0,
    background_phi_smooth: float = 0.60,
    background_theta_smooth: float = 0.20,
    modality_weights: str = "2:1",
) -> int:
    """Обучает BigARTM для эксперимента. Возвращает 0 при успехе."""
    output_dir = artifacts_root / experiment / ("k%d" % k_domain)
    batches_dir = artifacts_root / experiment / "artm" / "batches"
    dictionary_path = artifacts_root / experiment / "artm" / "dictionary.dict"
    if not batches_dir.exists() or not any(batches_dir.iterdir()):
        return 1
    if not dictionary_path.exists():
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_names = ["topic_%03d" % i for i in range(k_domain)]
    background_names = ["background_%02d" % i for i in range(k_background)]
    all_topics = topic_names + background_names
    batch_vectorizer = artm.BatchVectorizer(data_path=str(batches_dir), data_format="batches")
    dictionary = artm.Dictionary()
    dictionary.load(str(dictionary_path))
    experiment_dir = artifacts_root / experiment
    available_modalities = get_available_modalities(dictionary, experiment_dir)
    class_ids_dict = parse_modality_weights(modality_weights)
    class_ids_dict = {k: v for k, v in class_ids_dict.items() if k in available_modalities}
    if not class_ids_dict:
        class_ids_dict = {"text": 1.0}
    model, seed_applied, modality_class_ids = build_model(
        all_topics,
        dictionary,
        theta_sparse=theta_sparse,
        phi_sparse=phi_sparse,
        decorrelator=decorrelator,
        seed=seed,
        background_names=background_names,
        background_phi_smooth=background_phi_smooth,
        background_theta_smooth=background_theta_smooth,
        class_ids=class_ids_dict,
    )
    for _ in range(passes):
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)
    try:
        model.dump(str(output_dir / "model.dump"))
    except AttributeError:
        model.save(str(output_dir / "model"))
    def score_values(name):
        tracker = model.score_tracker.get(name)
        if tracker is None:
            return {"last": None, "values": []}
        values = getattr(tracker, "value", None) or []
        return {"last": tracker.last_value, "values": list(values)}
    top_tokens_all = model.score_tracker.get("top_tokens")
    top_tokens_all = top_tokens_all.last_tokens if top_tokens_all else {}
    modality_mapping = {"text": "unigrams", "bigrams": "bigrams", "trigrams": "trigrams"}
    top_tokens_by_modality = {}
    for cid in modality_class_ids:
        score_name = "top_tokens_%s" % cid
        if score_name in model.score_tracker:
            top_tokens_by_modality[modality_mapping.get(cid, cid)] = model.score_tracker[score_name].last_tokens
    scores = {
        "perplexity": score_values("perplexity"),
        "sparsity_theta": score_values("sparsity_theta"),
        "sparsity_phi": score_values("sparsity_phi"),
        "top_words": top_tokens_all,
        "top_words_by_modality": top_tokens_by_modality,
        "k_domain": k_domain,
        "k_background": k_background,
        "passes": passes,
        "theta_sparse": theta_sparse,
        "phi_sparse": phi_sparse,
        "decorrelator": decorrelator,
        "background_phi_smooth": background_phi_smooth,
        "background_theta_smooth": background_theta_smooth,
        "seed": seed,
        "seed_applied": seed_applied,
        "modality_weights": class_ids_dict,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_sanitize_json(scores), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for mod_name, tokens_dict in top_tokens_by_modality.items():
        fname = "top_terms_%s.json" % mod_name
        (output_dir / fname).write_text(json.dumps(tokens_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(modality_class_ids) > 1:
        (output_dir / "modality_weights.json").write_text(
            json.dumps(class_ids_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0
