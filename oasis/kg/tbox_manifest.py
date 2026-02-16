# TBox manifest из ontology.owl.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, RDF
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDFS, SKOS


def expand_class_expr(g: Graph, node: Any) -> list[str]:
    if isinstance(node, BNode):
        union = next(g.objects(node, OWL.unionOf), None)
        if union:
            return [str(x) for x in Collection(g, union)]
    return [str(node)]


def load_tbox_manifest(tbox_path: Path | str) -> dict:
    g = Graph()
    g.parse(str(tbox_path))
    classes = []
    object_props = []
    data_props = []
    subclass_map = defaultdict(set)
    label_index = defaultdict(list)
    domain_range_index = {}

    for cls in g.subjects(RDF.type, OWL.Class):
        labels = [str(l) for l in g.objects(cls, RDFS.label)]
        alt_labels = [str(l) for l in g.objects(cls, SKOS.altLabel)]
        entry = {"iri": str(cls), "labels": labels, "alt_labels": alt_labels}
        classes.append(entry)
        for label in labels + alt_labels:
            label_index[label.lower().strip()].append(str(cls))

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        labels = [str(l) for l in g.objects(prop, RDFS.label)]
        alt_labels = [str(l) for l in g.objects(prop, SKOS.altLabel)]
        raw_domains = list(g.objects(prop, RDFS.domain))
        raw_ranges = list(g.objects(prop, RDFS.range))
        domains = []
        ranges = []
        for node in raw_domains:
            domains.extend(expand_class_expr(g, node))
        for node in raw_ranges:
            ranges.extend(expand_class_expr(g, node))
        object_props.append({
            "iri": str(prop),
            "labels": labels,
            "alt_labels": alt_labels,
            "domain": domains,
            "range": ranges,
        })
        domain_range_index[str(prop)] = {"domain": domains, "range": ranges}
        for label in labels + alt_labels:
            label_index[label.lower().strip()].append(str(prop))

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        labels = [str(l) for l in g.objects(prop, RDFS.label)]
        domains = [str(o) for o in g.objects(prop, RDFS.domain)]
        ranges = [str(o) for o in g.objects(prop, RDFS.range)]
        data_props.append({"iri": str(prop), "labels": labels, "domain": domains, "range": ranges})
        domain_range_index[str(prop)] = {"domain": domains, "range": ranges}
        for label in labels:
            label_index[label.lower().strip()].append(str(prop))

    for child, parent in g.subject_objects(RDFS.subClassOf):
        subclass_map[str(child)].add(str(parent))

    return {
        "classes": classes,
        "object_properties": object_props,
        "datatype_properties": data_props,
        "label_index": {k: sorted(set(v)) for k, v in label_index.items()},
        "domain_range_index": domain_range_index,
        "subclass_map": {k: sorted(v) for k, v in subclass_map.items()},
    }
