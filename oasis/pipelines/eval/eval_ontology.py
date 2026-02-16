# Метрики OWL-онтологии (размер, структура, согласованность). Вызывается из oasis.pipelines.eval.run.

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rdflib import BNode, Graph, RDF, RDFS, OWL
from rdflib.namespace import SKOS
from tqdm import tqdm


def load_graph(path: Path) -> Graph:
    graph = Graph()
    graph.parse(path.as_posix())
    return graph


def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def split_camel(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    return text


def iter_classes(graph: Graph) -> set:
    classes = set(graph.subjects(RDF.type, OWL.Class))
    classes.update(graph.subjects(RDF.type, RDFS.Class))
    return classes


def iter_object_properties(graph: Graph) -> set:
    return set(graph.subjects(RDF.type, OWL.ObjectProperty))


def iter_data_properties(graph: Graph) -> set:
    return set(graph.subjects(RDF.type, OWL.DatatypeProperty))


def iter_annotation_properties(graph: Graph) -> set:
    return set(graph.subjects(RDF.type, OWL.AnnotationProperty))


def iter_individuals(graph: Graph) -> set:
    return set(graph.subjects(RDF.type, OWL.NamedIndividual))


def compute_depths(classes: set, edges: dict) -> dict:
    depths: dict = {}
    visiting: set = set()

    def dfs(node: Any) -> int:
        if node in depths:
            return depths[node]
        if node in visiting:
            return 0
        visiting.add(node)
        parents = edges.get(node, set())
        depth = 1 + max(dfs(p) for p in parents) if parents else 0
        visiting.remove(node)
        depths[node] = depth
        return depth

    for cls in classes:
        dfs(cls)
    return depths


def label_coverage(resources: Any, graph: Graph, *, top_n: int) -> dict:
    missing = []
    total = 0
    with_label = 0
    for res in resources:
        total += 1
        has_label = any(True for _ in graph.objects(res, RDFS.label))
        if has_label:
            with_label += 1
        elif len(missing) < top_n:
            missing.append(str(res))
    return {
        "total": total,
        "with_label": with_label,
        "missing_ratio": ((total - with_label) / total) if total else 0.0,
        "missing_examples": missing,
    }


def run(
    input_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    reasoner: str = "pellet",
    log_level: str = "INFO",
) -> None:
    """Считает метрики онтологии и пишет output_path (JSON)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config: dict = {}
    if config_path and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ont_cfg = config.get("ontology", {})
    top_n = ont_cfg.get("top_n", 20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading ontology from {}", input_path)
    graph = load_graph(input_path)
    logger.info("Loaded {} triples", len(graph))

    classes = iter_classes(graph)
    object_properties = iter_object_properties(graph)
    data_properties = iter_data_properties(graph)
    annotation_properties = iter_annotation_properties(graph)
    individuals = iter_individuals(graph)

    subclass_edges: dict = defaultdict(set)
    for cls in tqdm(classes, desc="Scanning subclasses"):
        for parent in graph.objects(cls, RDFS.subClassOf):
            if isinstance(parent, BNode):
                continue
            if parent in classes:
                subclass_edges[cls].add(parent)

    depths = compute_depths(classes, subclass_edges)
    depth_values = list(depths.values())
    subclass_edges_count = sum(len(parents) for parents in subclass_edges.values())
    children_map: dict = defaultdict(set)
    for child, parents in subclass_edges.items():
        for parent in parents:
            children_map[parent].add(child)
    branching_factors = [len(children_map.get(cls, set())) for cls in classes]
    avg_branching_factor = (
        (sum(branching_factors) / len(branching_factors)) if branching_factors else 0.0
    )
    roots = [
        cls
        for cls in classes
        if not subclass_edges.get(cls) and str(cls) != str(OWL.Thing)
    ]

    annotation_predicates = {RDFS.label, RDFS.comment, SKOS.prefLabel}
    n_annotations = sum(
        1 for _ in graph.triples((None, None, None)) if _[1] in annotation_predicates
    )

    dangling_classes = []
    dangling_classes_total = 0
    for cls in classes:
        has_relations = (
            bool(subclass_edges.get(cls))
            or any(True for _ in graph.subjects(RDFS.subClassOf, cls))
            or any(True for _ in graph.objects(cls, None))
        )
        if not has_relations:
            dangling_classes_total += 1
            if len(dangling_classes) < top_n:
                dangling_classes.append(str(cls))

    orphan_properties = []
    orphan_properties_total = 0
    for prop in list(object_properties) + list(data_properties):
        has_domain = any(True for _ in graph.objects(prop, RDFS.domain))
        has_range = any(True for _ in graph.objects(prop, RDFS.range))
        if has_domain or has_range:
            continue
        if any(True for _ in graph.triples((None, prop, None))):
            continue
        orphan_properties_total += 1
        if len(orphan_properties) < top_n:
            orphan_properties.append(str(prop))

    label_to_uris: dict = defaultdict(list)
    for entity in list(classes) + list(object_properties) + list(data_properties):
        for label in graph.objects(entity, RDFS.label):
            label_text = str(label).strip().lower()
            if label_text:
                label_to_uris[label_text].append(str(entity))
    label_collisions = []
    label_collisions_total = 0
    for label_text, uris in label_to_uris.items():
        if len(uris) > 1:
            label_collisions_total += 1
            label_collisions.append({"label": label_text, "uris": uris})
    label_collisions = label_collisions[:top_n]

    bad_name_re = re.compile(r"^(class|property|entity|concept)[_\-]?\d+$")
    bad_named = 0
    total_named = 0
    for entity in list(classes) + list(object_properties) + list(data_properties):
        label = None
        for cand in graph.objects(entity, RDFS.label):
            label = str(cand).strip()
            break
        if not label:
            label = split_camel(local_name(str(entity)))
        if not label:
            continue
        total_named += 1
        if bad_name_re.match(label.strip().lower()):
            bad_named += 1

    label_metrics = {
        "classes": label_coverage(tqdm(classes, desc="Labels: classes"), graph, top_n=top_n),
        "object_properties": label_coverage(
            tqdm(object_properties, desc="Labels: obj props"), graph, top_n=top_n
        ),
        "data_properties": label_coverage(
            tqdm(data_properties, desc="Labels: data props"), graph, top_n=top_n
        ),
        "annotation_properties": label_coverage(
            tqdm(annotation_properties, desc="Labels: anno props"), graph, top_n=top_n
        ),
    }

    entities_total = len(classes) + len(object_properties) + len(data_properties)
    entities_with_annotations = 0
    for entity in list(classes) + list(object_properties) + list(data_properties):
        has_label = any(True for _ in graph.objects(entity, RDFS.label))
        has_comment = any(True for _ in graph.objects(entity, RDFS.comment))
        has_pref = any(True for _ in graph.objects(entity, SKOS.prefLabel))
        if has_label or has_comment or has_pref:
            entities_with_annotations += 1

    try:
        import rdflib
        rdflib_version = getattr(rdflib, "__version__", None)
    except Exception:
        rdflib_version = None

    structure = {
        "subclass_edges_count": subclass_edges_count,
        "max_depth": max(depth_values) if depth_values else 0,
        "avg_depth": (sum(depth_values) / len(depth_values)) if depth_values else 0.0,
        "avg_branching_factor": avg_branching_factor,
        "roots_count": len(roots),
    }
    metrics = {
        "meta": {
            "config_path": str(config_path) if config_path and config_path.exists() else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "versions": {"rdflib": rdflib_version},
        },
        "size": {
            "triples": len(graph),
            "classes": len(classes),
            "object_properties": len(object_properties),
            "data_properties": len(data_properties),
            "annotation_properties": len(annotation_properties),
            "individuals": len(individuals),
            "axioms_proxy": len(graph),
            "annotations": n_annotations,
        },
        "structure": structure,
        "subclass_depth": {"max_depth": structure["max_depth"], "avg_depth": structure["avg_depth"]},
        "richness": {
            "relationship_richness": (
                len(object_properties) / (len(object_properties) + len(data_properties))
                if (len(object_properties) + len(data_properties)) else 0.0
            ),
            "attribute_richness": (len(data_properties) / len(classes)) if classes else 0.0,
            "inheritance_richness": (subclass_edges_count / len(classes)) if classes else 0.0,
            "annotation_richness": (
                (entities_with_annotations / entities_total) if entities_total else 0.0
            ),
        },
        "sanity_checks": {
            "dangling_classes_count": dangling_classes_total,
            "dangling_classes_examples": dangling_classes,
            "orphan_properties_count": orphan_properties_total,
            "orphan_properties_examples": orphan_properties,
            "label_collision_count": label_collisions_total,
            "label_collision_examples": label_collisions,
            "bad_naming_ratio": (bad_named / total_named) if total_named else 0.0,
        },
        "labels": label_metrics,
    }

    logger.info("Running consistency check with reasoner={}", reasoner)
    consistency: dict = {"status": "error", "reason": "unknown"}
    try:
        from owlready2 import World, sync_reasoner, sync_reasoner_pellet
        world = World()
        ont = world.get_ontology(input_path.as_posix()).load()
        if reasoner == "pellet":
            sync_reasoner_pellet(
                world,
                infer_property_values=True,
                infer_data_property_values=True,
                debug=0,
            )
        else:
            sync_reasoner(
                world,
                infer_property_values=True,
                infer_data_property_values=True,
                debug=0,
            )
        inconsistent = list(world.inconsistent_classes())
        consistency = {
            "status": "ok",
            "is_consistent": len(inconsistent) == 0,
            "unsatisfiable_classes_count": len(inconsistent),
            "unsatisfiable_classes_top": [str(cls) for cls in inconsistent[:top_n]],
        }
    except Exception as exc:
        logger.error("Consistency check failed: {}", exc)
        consistency = {"status": "error", "reason": str(exc)}
    metrics["consistency"] = consistency

    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote metrics to {}", output_path)
