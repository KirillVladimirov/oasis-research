# Материализация KG из ontology.owl и cqs_grounded.jsonl.

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from loguru import logger
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS
from rdflib.namespace import OWL, PROV
from tqdm import tqdm

from oasis.kg.tbox_manifest import load_tbox_manifest


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_]", "_", (s or "").strip())[:200]


def _doc_id_from_chunk_id(chunk_id: str) -> str:
    if "::" in chunk_id:
        return chunk_id.split("::")[0].strip()
    return chunk_id.strip()


def _load_cqs_grounded(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_references(data_root: Path, dataset: str) -> dict[str, dict]:
    refs_path = data_root / dataset / "references.csv"
    if not refs_path.exists():
        return {}
    refs = {}
    with refs_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("ref_id") or Path(row.get("pdf_url", "")).stem
            if not pid:
                continue
            refs[pid] = {
                "title": row.get("title", ""),
                "year": int(row["year"]) if row.get("year") else None,
                "venue": row.get("venue", ""),
            }
    return refs


def run_kg_materialize(
    artifacts_dir: Path,
    dataset: str,
    config: dict,
    data_root: Path | None = None,
) -> int:
    """Строит KG из ontology.owl и cqs_grounded.jsonl. Пишет kg/kg.ttl, kg_provenance, kg_stats.json."""
    base_iri = (config.get("ontology") or {}).get("base_iri", "http://example.org")
    base_iri = base_iri.rstrip("#/")
    kg_ns = Namespace(base_iri + "/kg/")

    ontology_path = artifacts_dir / dataset / "ontology.owl"
    cqs_path = artifacts_dir / dataset / "cqs_grounded.jsonl"
    kg_dir = artifacts_dir / dataset / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)

    if not ontology_path.exists():
        logger.error("Не найден: {}", ontology_path)
        return 1
    if not cqs_path.exists():
        logger.error("Не найден: {}", cqs_path)
        return 1

    logger.info("KG: dataset={}, out={}", dataset, kg_dir)
    tbox = load_tbox_manifest(ontology_path)
    g = Graph()
    g.parse(str(ontology_path))
    g.bind("kg", kg_ns)
    g.bind("prov", PROV)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    type_paper = kg_ns.Paper
    type_chunk = kg_ns.Chunk
    type_cq = kg_ns.CQ
    prop_supported_by = kg_ns.supportedBy
    prop_in_paper = kg_ns.inPaper
    prop_chunk_id = kg_ns.chunkId
    prop_doc_id = kg_ns.docId
    prop_text = kg_ns.text
    prop_text_span = kg_ns.textSpan
    type_support = kg_ns.Support
    prop_from_cq = kg_ns.fromCQ
    prop_to_chunk = kg_ns.toChunk
    prop_score = kg_ns.score

    papers: set[str] = set()
    chunks: set[str] = set()
    seen_in_paper: set[tuple[str, str]] = set()
    provenance_rows: list[dict] = []
    node_counts = {"Paper": 0, "Chunk": 0, "CQ": 0}
    edge_counts = {"supportedBy": 0, "inPaper": 0}

    refs = {}
    if data_root:
        refs = _load_references(data_root, dataset)

    cqs = _load_cqs_grounded(cqs_path)
    logger.info("CQ строк: {}", len(cqs))
    for cq_row in tqdm(cqs, desc="CQ", unit="cq"):
        cq_id = (cq_row.get("cq_id") or "").strip()
        if not cq_id:
            continue
        cq_text = (cq_row.get("text") or "").strip()
        cq_uri = kg_ns["cq/%s" % _slug(cq_id)]
        if (cq_uri, RDF.type, type_cq) not in g:
            g.add((cq_uri, RDF.type, type_cq))
            g.add((cq_uri, kg_ns.cqId, Literal(cq_id)))
            if cq_text:
                g.add((cq_uri, prop_text, Literal(cq_text)))
            node_counts["CQ"] += 1

        for ev in cq_row.get("evidence_chunks") or []:
            chunk_id = (ev.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            doc_id = ev.get("article_id") or ev.get("paper_id") or _doc_id_from_chunk_id(chunk_id)
            score = ev.get("score")
            if score is None:
                score = 0.0
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            papers.add(doc_id)
            chunks.add(chunk_id)

            chunk_uri = kg_ns["chunk/%s" % _slug(chunk_id)]
            if (chunk_uri, RDF.type, type_chunk) not in g:
                g.add((chunk_uri, RDF.type, type_chunk))
                g.add((chunk_uri, prop_chunk_id, Literal(chunk_id)))
                g.add((chunk_uri, prop_doc_id, Literal(doc_id)))
                node_counts["Chunk"] += 1
            text_span = (ev.get("text") or "")[:500]
            if text_span:
                g.add((chunk_uri, prop_text_span, Literal(text_span)))

            paper_uri = kg_ns["paper/%s" % _slug(doc_id)]
            if (paper_uri, RDF.type, type_paper) not in g:
                g.add((paper_uri, RDF.type, type_paper))
                g.add((paper_uri, prop_doc_id, Literal(doc_id)))
                meta = refs.get(doc_id, {})
                if meta.get("title"):
                    g.add((paper_uri, kg_ns.title, Literal(meta["title"])))
                if meta.get("year") is not None:
                    g.add((paper_uri, kg_ns.year, Literal(meta["year"])))
                if meta.get("venue"):
                    g.add((paper_uri, kg_ns.venue, Literal(meta["venue"])))
                node_counts["Paper"] += 1

            if (chunk_id, doc_id) not in seen_in_paper:
                seen_in_paper.add((chunk_id, doc_id))
                g.add((chunk_uri, prop_in_paper, paper_uri))
                edge_counts["inPaper"] += 1

            support = BNode()
            g.add((support, RDF.type, type_support))
            g.add((support, prop_from_cq, cq_uri))
            g.add((support, prop_to_chunk, chunk_uri))
            g.add((support, prop_score, Literal(score)))
            g.add((support, PROV.wasDerivedFrom, chunk_uri))
            g.add((cq_uri, prop_supported_by, chunk_uri))
            edge_counts["supportedBy"] += 1

            provenance_rows.append({
                "subject": str(cq_uri),
                "predicate": str(prop_supported_by),
                "object": str(chunk_uri),
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "cq_id": cq_id,
            })

    ttl_path = kg_dir / "kg.ttl"
    g.serialize(destination=str(ttl_path), format="turtle")
    logger.info("Записано {} (триплов: {})", ttl_path, len(g))
    # Smoke-check 9.5.6: граф парсится обратно; у всех Support есть prov:wasDerivedFrom
    try:
        g2 = Graph()
        g2.parse(str(ttl_path), format="turtle")
        support_nodes = list(g2.subjects(RDF.type, type_support))
        without_evidence = [s for s in support_nodes if not list(g2.objects(s, PROV.wasDerivedFrom))]
        if without_evidence:
            logger.warning("Support без evidence: {}", len(without_evidence))
        else:
            logger.info("Smoke-check: {} Support-узлов, у всех есть prov:wasDerivedFrom", len(support_nodes))
    except Exception as e:
        logger.warning("Smoke-check не выполнен: {}", e)

    if provenance_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(provenance_rows)
            df.to_parquet(kg_dir / "kg_provenance.parquet", index=False)
            logger.info("Записано kg_provenance.parquet (строк: {})", len(provenance_rows))
        except Exception as e:
            csv_path = kg_dir / "kg_provenance.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(provenance_rows[0].keys()))
                w.writeheader()
                w.writerows(provenance_rows)
            logger.warning("Parquet недоступен ({}), записан {}", e, csv_path)

    class_iris = [c["iri"] for c in tbox.get("classes", [])]
    rel_iris = [p["iri"] for p in tbox.get("object_properties", [])]
    def _first_label(obj: dict, fallback_key: str = "iri") -> str:
        labels = obj.get("labels") or [obj.get(fallback_key, "")]
        return (labels or [""])[0]

    top_relations = [_first_label(p) for p in tbox.get("object_properties", [])[:20]]
    top_classes = [_first_label(c) for c in tbox.get("classes", [])[:20]]
    stats = {
        "nodes": {
            "Paper": node_counts["Paper"],
            "Chunk": node_counts["Chunk"],
            "CQ": node_counts["CQ"],
            "Class_tbox": len(class_iris),
            "Relation_tbox": len(rel_iris),
        },
        "edges": edge_counts,
        "triples": len(g),
        "top_relations": top_relations,
        "top_classes": top_classes,
        "documents_covered": len(papers),
        "chunks_covered": len(chunks),
    }
    stats_path = kg_dir / "kg_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Записано {}", stats_path)
    logger.info("Итог: Paper={}, Chunk={}, CQ={}, supportedBy={}, inPaper={}, документов={}",
                node_counts["Paper"], node_counts["Chunk"], node_counts["CQ"],
                edge_counts["supportedBy"], edge_counts["inPaper"], len(papers))
    return 0
