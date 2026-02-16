# Оценка качества KG (Step 11): метрики, SHACL, единый CSV по плану 11.x.

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from loguru import logger
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import PROV

from oasis.kg.tbox_manifest import load_tbox_manifest

# Контракт колонок единого CSV (11.3)
KG_METRICS_CSV_COLUMNS = [
    "dataset_id",
    "kg_path",
    "run_ts",
    "parse_ok",
    "shacl_ran",
    "shacl_violations",
    "n_triples",
    "n_statements",
    "n_entities",
    "typed_ratio",
    "n_papers",
    "n_chunks",
    "n_cqs",
    "evidence_coverage",
    "avg_evidence_per_statement",
    "doc_diversity",
    "doc_top1_share",
    "doc_entropy",
    "n_components",
    "giant_component_share",
    "avg_degree",
    "predicate_top1_share",
    "predicate_top3_share",
    "predicate_entropy",
    "runtime_sec",
    "error_msg",
    "status",
]


def _kg_ns(base_iri: str):
    base = (base_iri or "http://example.org").rstrip("#/")
    return Namespace(base + "/kg/")


def load_kg(kg_path: Path, ontology_path: Path | None, base_iri: str) -> tuple[Graph, dict[str, Any] | None]:
    """Загружает kg.ttl и опционально ontology.owl. Возвращает (graph, tbox_manifest)."""
    g = Graph()
    g.parse(str(kg_path), format="turtle")
    tbox = None
    if ontology_path and ontology_path.exists():
        tbox = load_tbox_manifest(ontology_path)
    return g, tbox


def _load_provenance_file(prov_path: Path | None) -> dict[str, str]:
    """Читает kg_provenance.csv или .parquet; возвращает chunk_id -> doc_id для ускорения."""
    if not prov_path or not prov_path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        if prov_path.suffix.lower() == ".csv":
            with prov_path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    cid = (row.get("chunk_id") or "").strip()
                    did = (row.get("doc_id") or "").strip()
                    if cid:
                        out[cid] = did
        else:
            import pandas as pd
            df = pd.read_parquet(prov_path)
            if "chunk_id" in df.columns and "doc_id" in df.columns:
                for _, row in df.iterrows():
                    cid = str(row.get("chunk_id", "")).strip()
                    did = str(row.get("doc_id", "")).strip()
                    if cid:
                        out[cid] = did
    except Exception as e:
        logger.warning("Не удалось загрузить provenance {}: {}", prov_path, e)
    return out


def rdf_stats(g: Graph) -> dict[str, int]:
    """Базовая статистика RDF: triples, subjects (resources), literals, blank nodes."""
    resources = set()
    literals_count = 0
    bnodes = set()
    for s, p, o in g:
        resources.add(s)
        resources.add(p)
        if isinstance(o, Literal):
            literals_count += 1
        else:
            resources.add(o)
            if hasattr(o, "identifier") and str(o).startswith("_"):
                bnodes.add(o)
        if hasattr(s, "identifier") and str(s).startswith("_"):
            bnodes.add(s)
    return {
        "triples": len(g),
        "resources": len(resources),
        "literals": literals_count,
        "blank_nodes": len(bnodes),
    }


def _support_nodes_and_provenance(
    g: Graph,
    kg_ns: Namespace,
    doc_id_from_file: dict[str, str] | None = None,
) -> tuple[list[dict], list[str], dict[str, float], dict[str, str]]:
    """Извлекает Support-узлы; doc_id из опционального prov-файла или из графа."""
    doc_id_from_file = doc_id_from_file or {}
    type_support = kg_ns.Support
    prop_from_cq = kg_ns.fromCQ
    prop_to_chunk = kg_ns.toChunk
    prop_score = kg_ns.score
    prop_doc_id = kg_ns.docId
    prop_chunk_id = kg_ns.chunkId
    support_list = []
    chunk_ids = []
    score_by_support: dict[str, float] = {}
    doc_id_by_chunk: dict[str, str] = {}
    for s in g.subjects(RDF.type, type_support):
        from_cq = list(g.objects(s, prop_from_cq))
        to_chunk = list(g.objects(s, prop_to_chunk))
        scores = list(g.objects(s, prop_score))
        prov_chunks = list(g.objects(s, PROV.wasDerivedFrom))
        chunk_uri = (to_chunk or prov_chunks or [None])[0]
        if chunk_uri is None:
            continue
        chunk_id = ""
        for o in g.objects(chunk_uri, prop_chunk_id):
            chunk_id = str(o.value) if isinstance(o, Literal) else str(o)
            break
        if not chunk_id:
            chunk_id = str(chunk_uri).split("/")[-1]
        doc_id = doc_id_from_file.get(chunk_id, "")
        if not doc_id and chunk_uri:
            for o in g.objects(chunk_uri, prop_doc_id):
                doc_id = str(o) if not isinstance(o, Literal) else str(o.value)
                break
        if chunk_uri:
            doc_id_by_chunk[str(chunk_uri)] = doc_id
        score = 0.0
        if scores:
            try:
                score = float(scores[0])
            except (TypeError, ValueError):
                pass
        score_by_support[str(s)] = score
        support_list.append({
            "support_id": str(s),
            "cq_uri": str(from_cq[0]) if from_cq else "",
            "chunk_uri": str(chunk_uri),
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "score": score,
            "has_prov": len(prov_chunks) > 0,
        })
        if chunk_id:
            chunk_ids.append(chunk_id)
    return support_list, chunk_ids, score_by_support, doc_id_by_chunk


def entity_graph_edges(g: Graph, kg_ns: Namespace) -> list[tuple[str, str, str]]:
    """Рёбра для графа сущностей: (source_uri, target_uri, predicate_local_name)."""
    type_paper = kg_ns.Paper
    type_chunk = kg_ns.Chunk
    type_cq = kg_ns.CQ
    prop_in_paper = kg_ns.inPaper
    prop_supported_by = kg_ns.supportedBy
    edges = []
    for s, p, o in g:
        if p == prop_in_paper and (o, RDF.type, type_paper) in g:
            edges.append((str(s), str(o), "inPaper"))
        if p == prop_supported_by and (s, RDF.type, type_cq) in g:
            edges.append((str(s), str(o), "supportedBy"))
    return edges


def schema_metrics(tbox: dict | None) -> dict[str, Any]:
    """Метрики схемы (TBox): классы, свойства, глубина, relationship/inheritance richness."""
    if not tbox:
        return {}
    classes = tbox.get("classes", [])
    obj_props = tbox.get("object_properties", [])
    data_props = tbox.get("datatype_properties", [])
    subclass_map = tbox.get("subclass_map", {})
    n_classes = len(classes)
    n_obj = len(obj_props)
    n_data = len(data_props)
    all_props = n_obj + n_data
    relationship_richness = (n_obj / all_props) if all_props else 0.0
    parent_to_children: dict[str, list[str]] = {}
    for child, parents in subclass_map.items():
        for p in parents:
            parent_to_children.setdefault(p, []).append(child)
    n_with_children = len(parent_to_children)
    total_children = sum(len(v) for v in parent_to_children.values())
    inheritance_richness = (total_children / n_with_children) if n_with_children else 0.0
    depths = {}
    for cls in classes:
        iri = cls.get("iri", "")
        if iri not in depths:
            _compute_depth(iri, subclass_map, depths)
    max_depth = max(depths.values()) if depths else 0
    avg_depth = (sum(depths.values()) / len(depths)) if depths else 0.0
    return {
        "n_classes": n_classes,
        "n_object_properties": n_obj,
        "n_datatype_properties": n_data,
        "relationship_richness": round(relationship_richness, 4),
        "inheritance_richness": round(inheritance_richness, 4),
        "max_class_depth": max_depth,
        "avg_class_depth": round(avg_depth, 4),
    }


def _compute_depth(iri: str, subclass_map: dict, depths: dict) -> int:
    if iri in depths:
        return depths[iri]
    parents = subclass_map.get(iri, [])
    if not parents:
        depths[iri] = 0
        return 0
    d = 1 + max(_compute_depth(p, subclass_map, depths) for p in parents)
    depths[iri] = d
    return d


def kb_metrics(g: Graph, kg_ns: Namespace) -> dict[str, Any]:
    """Метрики наполнения: Paper, Chunk, CQ, Support; typed ratio."""
    type_paper = kg_ns.Paper
    type_chunk = kg_ns.Chunk
    type_cq = kg_ns.CQ
    type_support = kg_ns.Support
    n_paper = len(list(g.subjects(RDF.type, type_paper)))
    n_chunk = len(list(g.subjects(RDF.type, type_chunk)))
    n_cq = len(list(g.subjects(RDF.type, type_cq)))
    n_support = len(list(g.subjects(RDF.type, type_support)))
    total_typed = n_paper + n_chunk + n_cq + n_support
    total_nodes = total_typed
    typed_ratio = (total_typed / total_nodes) if total_nodes else 1.0
    return {
        "n_papers": n_paper,
        "n_chunks": n_chunk,
        "n_cqs": n_cq,
        "n_supports": n_support,
        "typed_ratio": round(typed_ratio, 4),
    }


def _weakly_connected_components(edges: list[tuple[str, str, str]]) -> tuple[list[set[str]], set[str]]:
    """BFS по неориентированным рёбрам для подсчёта компонент."""
    adj: dict[str, set[str]] = {}
    for src, tgt, _ in edges:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)
    visited = set()
    comps = []
    for node in adj:
        if node in visited:
            continue
        comp = set()
        stack = [node]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in adj.get(u, set()):
                if v not in visited:
                    stack.append(v)
        comps.append(comp)
    giant = max(comps, key=len) if comps else set()
    return comps, giant


def graph_metrics(edges: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Компоненты связности, плотность, степень, entropy предикатов."""
    nodes = set()
    pred_counts: dict[str, int] = {}
    for src, tgt, pred in edges:
        nodes.add(src)
        nodes.add(tgt)
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    n = len(nodes)
    m = len(edges)
    density = (m / (n * (n - 1))) if n > 1 else 0.0
    avg_degree = (2 * m / n) if n else 0.0
    total = sum(pred_counts.values())
    entropy = 0.0
    for c in pred_counts.values():
        if total > 0:
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
    top_preds = sorted(pred_counts.items(), key=lambda x: -x[1])[:5]
    top3_share = sum(c for _, c in top_preds[:3]) / total if total else 0.0
    top1_share = top_preds[0][1] / total if total and top_preds else 0.0
    comps, giant = _weakly_connected_components(edges)
    giant_share = len(giant) / n if n else 0.0
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 4),
        "predicate_entropy": round(entropy, 4),
        "top1_predicate_share": round(top1_share, 4),
        "top3_predicate_share": round(top3_share, 4),
        "n_components": len(comps),
        "giant_component_share": round(giant_share, 4),
    }


def provenance_metrics(
    support_list: list[dict],
    score_by_support: dict[str, float],
    doc_id_by_chunk: dict[str, str],
) -> dict[str, Any]:
    """Evidence coverage, avg evidence per statement, doc diversity, doc_top1_share, doc_entropy."""
    if not support_list:
        return {}
    from collections import Counter

    n_statements = len(support_list)
    with_prov = sum(1 for r in support_list if r.get("has_prov"))
    evidence_coverage = with_prov / n_statements
    avg_evidence_per_statement = with_prov / n_statements if n_statements else 0.0
    scores = [r.get("score", 0.0) for r in support_list]
    doc_ids = [
        (r.get("doc_id") or doc_id_by_chunk.get(r.get("chunk_uri", ""), "")).strip()
        for r in support_list
    ]
    doc_ids = [d for d in doc_ids if d]
    doc_counts = Counter(doc_ids)
    unique_docs = len(doc_counts)
    if doc_counts:
        top_doc, top_count = doc_counts.most_common(1)[0]
        doc_top1_share = top_count / n_statements
        total = sum(doc_counts.values())
        doc_entropy = 0.0
        for c in doc_counts.values():
            p = c / total
            if p > 0:
                doc_entropy -= p * math.log2(p)
    else:
        doc_top1_share = 0.0
        doc_entropy = 0.0
    score_above_05 = sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0
    score_above_07 = sum(1 for s in scores if s >= 0.7) / len(scores) if scores else 0.0
    score_above_09 = sum(1 for s in scores if s >= 0.9) / len(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "evidence_coverage": round(evidence_coverage, 4),
        "avg_evidence_per_statement": round(avg_evidence_per_statement, 4),
        "n_supports_with_prov": with_prov,
        "doc_diversity": unique_docs,
        "doc_top1_share": round(doc_top1_share, 4),
        "doc_entropy": round(doc_entropy, 4),
        "doc_concentration_top1": round(doc_top1_share, 4),
        "score_mean": round(avg_score, 4),
        "score_ge_05_ratio": round(score_above_05, 4),
        "score_ge_07_ratio": round(score_above_07, 4),
        "score_ge_09_ratio": round(score_above_09, 4),
    }


def run_shacl(kg_path: Path, shapes_path: Path | None, base_iri: str) -> tuple[int, str, int]:
    """Запуск pySHACL. Возвращает (violations_count, report_text, shacl_ran). shacl_ran=1 если прогон был, 0 если нет."""
    try:
        import pyshacl
    except ImportError:
        return -1, "pyshacl not installed", 0
    if shapes_path and shapes_path.exists():
        conforms, _, report_text = pyshacl.validate(str(kg_path), shacl_graph=str(shapes_path))
    else:
        conforms, _, report_text = pyshacl.validate(str(kg_path), shacl_graph=None)
    report_str = str(report_text) if report_text else ""
    shacl_ran = 1
    if conforms:
        return 0, report_str, shacl_ran
    count = report_str.count("Validation Result") or 1
    return max(1, count), report_str, shacl_ran


def compute_status(
    parse_ok: int,
    shacl_ran: int,
    shacl_violations: int | str,
    evidence_coverage: float,
    doc_top1_share: float,
    predicate_top1_share: float,
    *,
    fail_evidence_threshold: float = 0.99,
    warn_doc_concentration: float = 0.4,
    warn_predicate_concentration: float = 0.5,
) -> str:
    """OK / WARN / FAIL по правилам 11.5."""
    if parse_ok == 0:
        return "FAIL"
    viol = shacl_violations if isinstance(shacl_violations, int) else 0
    if shacl_ran and viol > 0:
        return "FAIL"
    if evidence_coverage < fail_evidence_threshold:
        return "FAIL"
    if doc_top1_share > warn_doc_concentration or predicate_top1_share > warn_predicate_concentration:
        return "WARN"
    return "OK"


def stoplight(
    evidence_coverage: float,
    shacl_violations: int,
    giant_component_share: float,
    top1_predicate_share: float,
    *,
    min_evidence: float = 0.99,
    max_violations: int = 0,
    min_giant: float = 0.5,
    max_top1: float = 0.95,
) -> dict[str, Any]:
    """Пороги пригодности: evidence, SHACL, связность, невырожденность предикатов."""
    shacl_ok = (shacl_violations <= max_violations) if shacl_violations >= 0 else True
    return {
        "evidence_ok": evidence_coverage >= min_evidence,
        "shacl_ok": shacl_ok,
        "giant_ok": giant_component_share >= min_giant,
        "predicate_ok": top1_predicate_share <= max_top1,
        "all_ok": (
            evidence_coverage >= min_evidence
            and shacl_ok
            and giant_component_share >= min_giant
            and top1_predicate_share <= max_top1
        ),
    }


def _na() -> str:
    """Значение NA для CSV."""
    return ""


def run_kg_eval(
    kg_path: Path,
    ontology_path: Path | None,
    base_iri: str,
    out_dir: Path,
    prov_path: Path | None = None,
    run_shacl_validation: bool = True,
    shapes_path: Path | None = None,
) -> dict[str, Any]:
    """
    Полный прогон оценки одного KG.
    Возвращает dict с ключом "row" (строка для kg_metrics_all.csv) и "combined" (для JSON).
    При ошибке парсинга: row с parse_ok=0, error_msg, остальное NA.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()
    row: dict[str, Any] = {c: _na() for c in KG_METRICS_CSV_COLUMNS}
    row["kg_path"] = str(kg_path.resolve())
    row["run_ts"] = run_ts

    try:
        g = Graph()
        g.parse(str(kg_path), format="turtle")
    except Exception as e:
        row["parse_ok"] = 0
        row["error_msg"] = str(e)[:500]
        row["runtime_sec"] = round(time.perf_counter() - t0, 2)
        row["status"] = "FAIL"
        row["shacl_ran"] = 0
        row["shacl_violations"] = _na()
        logger.warning("Parse failed for {}: {}", kg_path, e)
        return {"row": row, "combined": None}

    row["parse_ok"] = 1
    tbox = None
    if ontology_path and ontology_path.exists():
        tbox = load_tbox_manifest(ontology_path)
    kg_ns = _kg_ns(base_iri)
    doc_id_from_file = _load_provenance_file(prov_path)
    support_list, _, score_by_support, doc_id_by_chunk = _support_nodes_and_provenance(
        g, kg_ns, doc_id_from_file
    )
    edges = entity_graph_edges(g, kg_ns)
    rdf = rdf_stats(g)
    schema = schema_metrics(tbox)
    kb = kb_metrics(g, kg_ns)
    graph_m = graph_metrics(edges)
    prov_m = provenance_metrics(support_list, score_by_support, doc_id_by_chunk)

    shacl_violations: int | str = 0
    shacl_ran = 0
    shacl_report = ""
    if run_shacl_validation:
        shacl_violations, shacl_report, shacl_ran = run_shacl(kg_path, shapes_path, base_iri)
    else:
        shacl_ran = 0
        shacl_violations = -1
    if shacl_ran == 0 or shacl_violations < 0:
        row["shacl_violations"] = _na()
        row["shacl_ran"] = 0
    else:
        row["shacl_violations"] = shacl_violations
        row["shacl_ran"] = shacl_ran

    n_entities = (kb.get("n_papers", 0) or 0) + (kb.get("n_chunks", 0) or 0) + (kb.get("n_cqs", 0) or 0)
    ev_cov = prov_m.get("evidence_coverage", 0.0)
    doc_top1 = prov_m.get("doc_top1_share", 0.0)
    pred_top1 = graph_m.get("top1_predicate_share", 0.0)

    row["n_triples"] = rdf.get("triples", 0)
    row["n_statements"] = kb.get("n_supports", 0)
    row["n_entities"] = n_entities
    row["typed_ratio"] = kb.get("typed_ratio", 0.0)
    row["n_papers"] = kb.get("n_papers", 0)
    row["n_chunks"] = kb.get("n_chunks", 0)
    row["n_cqs"] = kb.get("n_cqs", 0)
    row["evidence_coverage"] = ev_cov
    row["avg_evidence_per_statement"] = prov_m.get("avg_evidence_per_statement", 0.0)
    row["doc_diversity"] = prov_m.get("doc_diversity", 0)
    row["doc_top1_share"] = doc_top1
    row["doc_entropy"] = prov_m.get("doc_entropy", 0.0)
    row["n_components"] = graph_m.get("n_components", 0)
    row["giant_component_share"] = graph_m.get("giant_component_share", 0.0)
    row["avg_degree"] = graph_m.get("avg_degree", 0.0)
    row["predicate_top1_share"] = graph_m.get("top1_predicate_share", 0.0)
    row["predicate_top3_share"] = graph_m.get("top3_predicate_share", 0.0)
    row["predicate_entropy"] = graph_m.get("predicate_entropy", 0.0)
    row["runtime_sec"] = round(time.perf_counter() - t0, 2)
    row["status"] = compute_status(
        row["parse_ok"], row["shacl_ran"], row["shacl_violations"],
        ev_cov, doc_top1, pred_top1,
    )

    sl = stoplight(ev_cov, int(shacl_violations) if isinstance(shacl_violations, int) else 0,
                  graph_m.get("giant_component_share", 0.0), pred_top1)
    combined = {
        "rdf_stats": rdf,
        "schema_metrics": schema,
        "kb_metrics": kb,
        "graph_metrics": graph_m,
        "provenance_metrics": prov_m,
        "shacl_violations": shacl_violations,
        "stoplight": sl,
    }
    (out_dir / "schema_metrics.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "kb_metrics.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "graph_metrics.json").write_text(
        json.dumps(graph_m, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "provenance_metrics.json").write_text(
        json.dumps(prov_m, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "rdf_stats.json").write_text(json.dumps(rdf, indent=2), encoding="utf-8")
    (out_dir / "shacl_report.txt").write_text(shacl_report or "(no report)", encoding="utf-8")
    (out_dir / "shacl_violations_count.txt").write_text(
        str(shacl_violations) if isinstance(shacl_violations, int) else "NA", encoding="utf-8"
    )
    (out_dir / "stoplight.json").write_text(json.dumps(sl, indent=2), encoding="utf-8")
    (out_dir / "all_metrics.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "KG eval: triples={}, supports={}, evidence_cov={}, status={}",
        rdf["triples"], kb.get("n_supports", 0), ev_cov, row["status"],
    )
    return {"row": row, "combined": combined}


def write_kg_metrics_all_csv(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    sorted_path: Path | None = None,
) -> None:
    """Пишет kg_metrics_all.csv с фиксированным порядком колонок (11.3). Опционально sorted по evidence_coverage, shacl_violations."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = KG_METRICS_CSV_COLUMNS
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if sorted_path and rows:
        def _sort_key(r: dict) -> tuple:
            ec = r.get("evidence_coverage")
            ec = float(ec) if ec not in ("", None) else 0.0
            sh = r.get("shacl_violations")
            sh = int(sh) if isinstance(sh, int) else (999 if sh == "" else 999)
            return (-ec, sh)
        sorted_rows = sorted(rows, key=_sort_key)
        with sorted_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted_rows)
        logger.info("Записано {} и {}", out_path, sorted_path)


def load_datasets_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Читает datasets_manifest.csv (11.1 Вариант А). Колонки: dataset_id, kg_path [, stats_path, prov_path]."""
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            did = (row.get("dataset_id") or "").strip()
            kg = (row.get("kg_path") or "").strip()
            if did and kg:
                rows.append({
                    "dataset_id": did,
                    "kg_path": kg,
                    "stats_path": (row.get("stats_path") or "").strip(),
                    "prov_path": (row.get("prov_path") or "").strip(),
                })
    return rows
