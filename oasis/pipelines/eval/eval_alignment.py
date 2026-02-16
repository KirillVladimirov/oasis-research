# Метрики выравнивания CQ и онтологии. Вызывается из oasis.pipelines.eval.run.

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS
from tqdm import tqdm

from oasis.pipelines.eval.cq_loader import load_cqs
from oasis.pipelines.eval.cq_normalize import normalize_cq, tokenize

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "the", "to", "with",
}

INTENT_PATTERNS = {
    "numeric": re.compile(r"\b(how many|how much|number of|amount of|average|mean|count)\b"),
    "causal": re.compile(r"\b(impact|effect|influence|cause|causal|lead to|result in)\b"),
    "temporal": re.compile(r"\b(when|time|duration|before|after|timeline|trend)\b"),
    "comparative": re.compile(r"\b(compare|difference|versus|vs\.|greater|less)\b"),
}


def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def split_camel(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    return text


def normalize_surface(text: str) -> tuple[str, list[str]]:
    normalized, _ = normalize_cq(text, remove_trailing_period=True)
    tokens = tokenize(normalized)
    return normalized, tokens


def tokens_in_sequence(tokens: list[str], sequence: list[str]) -> bool:
    if not tokens or len(tokens) > len(sequence):
        return False
    for start in range(len(sequence) - len(tokens) + 1):
        if list(sequence[start : start + len(tokens)]) == list(tokens):
            return True
    return False


def filtered_tokens(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def build_lexicon(graph: Graph) -> tuple[dict, dict]:
    entities: dict = {}
    surface_index: dict = defaultdict(set)

    def register(entity_uri: str, entity_type: str, surface: str) -> None:
        normalized, tokens = normalize_surface(surface)
        if not normalized:
            return
        entities.setdefault(entity_uri, {"type": entity_type, "surfaces": []})
        entities[entity_uri]["surfaces"].append({"text": normalized, "tokens": tokens})
        surface_index[normalized].add(entity_uri)

    classes = set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
    object_props = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    for cls in classes:
        uri = str(cls)
        register(uri, "class", local_name(uri))
        for label in graph.objects(cls, RDFS.label):
            register(uri, "class", str(label))
        for label in graph.objects(cls, SKOS.prefLabel):
            register(uri, "class", str(label))
        for comment in graph.objects(cls, RDFS.comment):
            register(uri, "class", str(comment))
    for prop in object_props:
        uri = str(prop)
        register(uri, "object_property", local_name(uri))
        for label in graph.objects(prop, RDFS.label):
            register(uri, "object_property", str(label))
        for label in graph.objects(prop, SKOS.prefLabel):
            register(uri, "object_property", str(label))
        for comment in graph.objects(prop, RDFS.comment):
            register(uri, "object_property", str(comment))
    for prop in data_props:
        uri = str(prop)
        register(uri, "data_property", local_name(uri))
        for label in graph.objects(prop, RDFS.label):
            register(uri, "data_property", str(label))
        for label in graph.objects(prop, SKOS.prefLabel):
            register(uri, "data_property", str(label))
        for comment in graph.objects(prop, RDFS.comment):
            register(uri, "data_property", str(comment))
    for uri, payload in list(entities.items()):
        for surface in list(payload["surfaces"]):
            split = split_camel(surface["text"])
            if split != surface["text"]:
                register(uri, payload["type"], split)
    return entities, surface_index


def match_entities(
    cq_tokens: list[str],
    entities: dict,
    stopwords: set[str],
) -> tuple[set[str], set[str]]:
    mentioned_classes: set[str] = set()
    mentioned_properties: set[str] = set()
    for uri, payload in entities.items():
        entity_type = payload["type"]
        for surface in payload["surfaces"]:
            tokens = surface["tokens"]
            if not tokens:
                continue
            if tokens_in_sequence(tokens, cq_tokens):
                if entity_type == "class":
                    mentioned_classes.add(uri)
                else:
                    mentioned_properties.add(uri)
                break
            filtered = filtered_tokens(tokens, stopwords)
            if len(filtered) >= 2 and set(filtered).issubset(set(cq_tokens)):
                if entity_type == "class":
                    mentioned_classes.add(uri)
                else:
                    mentioned_properties.add(uri)
                break
            if len(filtered) == 1 and len(filtered[0]) >= 5 and filtered[0] in cq_tokens:
                if entity_type == "class":
                    mentioned_classes.add(uri)
                else:
                    mentioned_properties.add(uri)
                break
    return mentioned_classes, mentioned_properties


def compute_entropy(counts: list[int]) -> float:
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def run(
    cqs_path: Path,
    ontology_path: Path,
    output_path: Path,
    grounding_path: Path,
    config_path: Path | None = None,
    log_level: str = "INFO",
    top_n: int = 10,
) -> None:
    """Считает метрики выравнивания CQ–онтология и пишет output_path + grounding_path."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config: dict = {}
    if config_path and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    align_cfg = config.get("alignment", {})
    top_n = align_cfg.get("top_n", top_n)
    stopwords = set(align_cfg.get("stopwords", [])) or STOPWORDS
    intent_cfg = align_cfg.get("intent_patterns", {})
    intent_patterns = (
        {k: re.compile(v, re.IGNORECASE) for k, v in intent_cfg.items()} if intent_cfg
        else INTENT_PATTERNS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading ontology from {}", ontology_path)
    graph = Graph()
    graph.parse(ontology_path.as_posix())
    logger.info("Loaded {} triples", len(graph))
    logger.info("Building signature lexicon")
    entities, _ = build_lexicon(graph)
    classes = {uri for uri, payload in entities.items() if payload["type"] == "class"}
    properties = {
        uri for uri, payload in entities.items()
        if payload["type"] in {"object_property", "data_property"}
    }
    data_properties = {
        uri for uri, payload in entities.items() if payload["type"] == "data_property"
    }
    top_level_classes = set()
    for cls in classes:
        parents = {
            str(parent)
            for parent in graph.objects(URIRef(cls), RDFS.subClassOf)
            if str(parent) in classes
        }
        if not parents and cls != str(OWL.Thing):
            top_level_classes.add(cls)

    logger.info("Loading CQs from {}", cqs_path)
    cqs = load_cqs(cqs_path)
    mentioned_classes_all: set[str] = set()
    mentioned_properties_all: set[str] = set()
    grounded_count = 0
    total_entities_per_cq = 0
    entity_counts: Counter = Counter()
    unsupported_intents: Counter = Counter()
    supported_count = 0

    with grounding_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(cqs, desc="Grounding CQs"):
            normalized_text, _ = normalize_cq(record.text, remove_trailing_period=True)
            cq_tokens = tokenize(normalized_text)
            mentioned_classes, mentioned_properties = match_entities(
                cq_tokens, entities, stopwords
            )
            grounded = bool(mentioned_classes or mentioned_properties)
            grounded_count += int(grounded)
            total_entities_per_cq += len(mentioned_classes) + len(mentioned_properties)
            mentioned_classes_all.update(mentioned_classes)
            mentioned_properties_all.update(mentioned_properties)
            for uri in mentioned_classes:
                entity_counts[(uri, "class")] += 1
            for uri in mentioned_properties:
                entity_counts[(uri, "property")] += 1
            intent_types = [
                intent for intent, pattern in intent_patterns.items()
                if pattern.search(normalized_text)
            ]
            supported = grounded
            if "numeric" in intent_types and not data_properties:
                supported = False
                unsupported_intents["numeric"] += 1
            if "causal" in intent_types and not mentioned_properties:
                supported = False
                unsupported_intents["causal"] += 1
            if "temporal" in intent_types and not mentioned_properties:
                supported = False
                unsupported_intents["temporal"] += 1
            if "comparative" in intent_types and not mentioned_properties:
                supported = False
                unsupported_intents["comparative"] += 1
            supported_count += int(supported)
            handle.write(
                json.dumps(
                    {
                        "cq_id": record.id,
                        "mentioned_classes": sorted(mentioned_classes),
                        "mentioned_properties": sorted(mentioned_properties),
                        "grounded": grounded,
                        "intent_types": intent_types,
                        "supported_by_ontology": supported,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    top_entities = [
        {"entity": uri, "type": etype, "count": count}
        for (uri, etype), count in entity_counts.most_common(top_n)
    ]
    mention_counts = [c for (_, _), c in entity_counts.items()]
    coverage_entropy = compute_entropy(mention_counts)
    coverage_entropy_norm = (
        coverage_entropy / math.log(len(mention_counts)) if len(mention_counts) > 1 else 0.0
    )
    uncovered_top_level = sorted(top_level_classes - mentioned_classes_all)
    uncovered_classes = sorted(classes - mentioned_classes_all)
    uncovered_properties = sorted(properties - mentioned_properties_all)

    metrics: dict = {
        "meta": {
            "config_path": str(config_path) if config_path and config_path.exists() else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "grounded_cq_ratio": (grounded_count / len(cqs)) if cqs else 0.0,
        "avg_entities_per_cq": (total_entities_per_cq / len(cqs)) if cqs else 0.0,
        "classes_covered_by_cqs_ratio": (len(mentioned_classes_all) / len(classes)) if classes else 0.0,
        "properties_covered_by_cqs_ratio": (
            len(mentioned_properties_all) / len(properties) if properties else 0.0
        ),
        "signature_coverage_classes": (len(mentioned_classes_all) / len(classes)) if classes else 0.0,
        "signature_coverage_properties": (
            len(mentioned_properties_all) / len(properties) if properties else 0.0
        ),
        "uncovered_top_level_classes": uncovered_top_level[:top_n],
        "uncovered_classes_examples": uncovered_classes[:top_n],
        "uncovered_properties_examples": uncovered_properties[:top_n],
        "coverage_entropy": coverage_entropy,
        "coverage_entropy_normalized": coverage_entropy_norm,
        "top10_most_mentioned_entities": top_entities,
        "answerability_proxy_rate": (supported_count / len(cqs)) if cqs else 0.0,
        "unsupported_intent_types": dict(unsupported_intents),
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote alignment metrics to {}", output_path)
    logger.info("Wrote CQ grounding to {}", grounding_path)
