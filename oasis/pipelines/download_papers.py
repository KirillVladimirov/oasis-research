# Скачивание PDF из README GitHub Awesome-репозиториев.
# Сохранение: data_root / topic_id / paper_pdfs/

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


@dataclass
class TopicConfig:
    """Один топик: репозиторий и опционально anchor секции в README."""
    topic_name: str
    topic_id: str
    url: str
    anchor: Optional[str] = None
    parse_config: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.parse_config is None:
            self.parse_config = {}


TOPICS_CONFIG: list[TopicConfig] = [
    TopicConfig("Deep Learning Neural Networks", "deep_learning_neural_networks",
                "https://github.com/hurshd0/must-read-papers-for-ml",
                anchor="books-neural-networks-and-deep-learning-neural-networks"),
    TopicConfig("ML Systems", "ml_systems_data_processing", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="data-processing"),
    TopicConfig("ML Systems", "ml_systems_rl_post_training", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="rl-post-training"),
    TopicConfig("ML Systems", "ml_systems_llm_long_context", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="llm-long-context"),
    TopicConfig("ML Systems", "ml_systems_mixture_of_experts_moe", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="mixture-of-experts-moe"),
    TopicConfig("ML Systems", "ml_systems_inference_system", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="inference-system"),
    TopicConfig("ML Systems", "ml_systems_distributed_training", "https://github.com/byungsoo-oh/ml-systems-papers", anchor="distributed-training"),
    TopicConfig("Model Quantization", "model_quantization", "https://github.com/Efficient-ML/Awesome-Model-Quantization"),
    TopicConfig("Multimodal AI", "multimodal_ai", "https://github.com/friedrichor/Awesome-Multimodal-Papers"),
    TopicConfig("Generative Models in Pathology", "generative_models_in_pathology", "https://github.com/yuanzhang7/Awesome-Generative-Models-in-Pathology"),
    TopicConfig("Graph NN Systems", "graph_nn_systems", "https://github.com/ch-wan/awesome-gnn-systems"),
    TopicConfig("Graph NN Research", "graph_nn_research", "https://github.com/xkLi-Allen/Awesome-GNN-Research"),
    TopicConfig("Speech Recognition & Synthesis", "speech_recognition_synthesis", "https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers"),
    TopicConfig("AutoML", "automl", "https://github.com/hibayesian/awesome-automl-papers"),
    TopicConfig("Healthcare Foundation Models", "healthcare_foundation_models", "https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models"),
    TopicConfig("Causal Inference", "causal_inference", "https://github.com/matthewvowels1/Awesome-Causal-Inference"),
    TopicConfig("Diffusion Models", "diffusion_models", "https://github.com/diff-usion/Awesome-Diffusion-Models"),
]


def get_github_raw_readme(repo_url: str, branch: str = "main") -> str:
    """Скачивает README.md репозитория (raw). Пробует branch и master."""
    m = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        raise ValueError("Неверный GitHub URL: %s" % repo_url)
    user, repo = m.groups()
    for b in [branch, "master"]:
        url = "https://raw.githubusercontent.com/%s/%s/%s/README.md" % (user, repo, b)
        r = requests.get(url, timeout=30)
        if r.ok:
            return r.text
    r.raise_for_status()


def normalize_heading_for_anchor(text: str) -> str:
    """Нормализация заголовка как в GitHub (для поиска по anchor)."""
    s = re.sub(r"[^\w\s-]", "", text.lower().strip())
    s = re.sub(r"[_\s]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def extract_section_by_anchor(markdown: str, anchor: str) -> str:
    """Вырезает из markdown блок от заголовка с данным anchor до следующего заголовка того же уровня."""
    lines = markdown.splitlines()
    start = level = None
    an = anchor.lower().strip()
    for i, line in enumerate(lines):
        h = re.match(r"^(#{1,6})\s+(.+)$", line)
        if h:
            lv = len(h.group(1))
            hn = normalize_heading_for_anchor(h.group(2).strip())
            if hn == an or hn.startswith(an + "-"):
                start, level = i, lv
                break
    if start is None:
        return ""
    out = []
    for i in range(start + 1, len(lines)):
        line = lines[i]
        h = re.match(r"^(#{1,6})\s+.+$", line)
        if h and len(h.group(1)) <= level:
            break
        out.append(line)
    return "\n".join(out)


def parse_markdown_links(markdown: str) -> list[str]:
    """Собирает URL из markdown: [[...]](url), [text](url) и голые https://..."""
    urls = []
    seen = set()
    used = set()
    for m in re.finditer(r"\[\[[^\]]+\]\]\(([^)]+)\)", markdown):
        u = m.group(1).strip()
        used.update(range(m.start(), m.end()))
        if u and not u.startswith("#") and u not in seen:
            urls.append(u)
            seen.add(u)
    for m in re.finditer(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)", markdown):
        if any(m.start() <= p < m.end() for p in used):
            continue
        u = m.group(1).strip()
        used.update(range(m.start(), m.end()))
        if u and not u.startswith("#") and u not in seen:
            urls.append(u)
            seen.add(u)
    for m in re.finditer(r"https?://[^\s\)\]>]+", markdown):
        if any(m.start() <= p < m.end() for p in used):
            continue
        u = m.group(0).rstrip(".,;:!?")
        if u not in seen:
            urls.append(u)
            seen.add(u)
    return urls


def identify_pdf_source(url: str) -> Optional[dict[str, str]]:
    """Определяет тип ссылки (arxiv, прямая PDF, openreview) и возвращает pdf_url и paper_id."""
    u = url.lower()
    for pat in [r"arxiv\.org/abs/(\d+\.\d+v?\d*)", r"arxiv\.org/pdf/(\d+\.\d+v?\d*)"]:
        m = re.search(pat, u)
        if m:
            aid = m.group(1)
            aid_plain = re.sub(r"v\d+$", "", aid)
            return {"type": "arxiv", "pdf_url": "https://arxiv.org/pdf/%s.pdf" % aid_plain, "paper_id": aid}
    if u.endswith(".pdf") or ".pdf?" in u or ".pdf#" in u:
        pid = url.split("/")[-1].split("?")[0].split("#")[0].replace(".pdf", "")
        if not pid or len(pid) < 3:
            pid = hashlib.md5(url.encode()).hexdigest()[:12]
        return {"type": "direct", "pdf_url": url, "paper_id": pid}
    if "openreview.net" in u:
        if "/pdf?id=" in u:
            pdf_url = url
        elif "/forum?id=" in u:
            pdf_url = url.replace("/forum?id=", "/pdf?id=")
        elif "?id=" in u:
            pdf_url = url.replace("?", "/pdf?id=", 1)
        else:
            return None
        pid = url.split("id=")[-1].split("&")[0].split("#")[0] if "id=" in url else hashlib.md5(url.encode()).hexdigest()[:12]
        return {"type": "openreview", "pdf_url": pdf_url, "paper_id": pid}
    return None


def download_pdf(pdf_info: dict[str, str], output_path: Path, max_retries: int = 3) -> bool:
    """Скачивает один PDF. Возвращает True при успехе или если файл уже есть."""
    if output_path.exists():
        return True
    url = pdf_info["pdf_url"]
    for attempt in range(max_retries):
        r = requests.get(url, timeout=60, stream=True)
        if r.ok:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"".join(r.iter_content(chunk_size=8192)))
            return True
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return False


def download_topic(config: TopicConfig, data_root: Path) -> dict[str, int]:
    """Скачивает все PDF по топику. Каталог: data_root / topic_id / paper_pdfs/.
    Возвращает счётчики: found, downloaded, skipped, errors."""
    stats = {"found": 0, "downloaded": 0, "skipped": 0, "errors": 0}
    branch = config.parse_config.get("branch", "main")
    markdown = get_github_raw_readme(config.url, branch=branch)
    if config.anchor:
        markdown = extract_section_by_anchor(markdown, config.anchor)
        if not markdown:
            return stats
    links = parse_markdown_links(markdown)
    pdf_infos = []
    for link in links:
        info = identify_pdf_source(link)
        if info:
            pdf_infos.append(info)
    stats["found"] = len(pdf_infos)
    if not pdf_infos:
        return stats

    out_dir = data_root / config.topic_id / "paper_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for info in tqdm(pdf_infos, desc=config.topic_id, unit="pdf"):
        pid = info["paper_id"]
        if info["type"] == "arxiv":
            name = "%s.pdf" % pid
        else:
            safe = re.sub(r'[<>:"/\\|?*]', "_", pid)
            if len(safe) > 200 or not safe.strip():
                name = "%s.pdf" % hashlib.md5(pid.encode()).hexdigest()[:12]
            else:
                name = "%s.pdf" % safe
        path = out_dir / name
        if path.exists():
            stats["skipped"] += 1
            continue
        if download_pdf(info, path):
            stats["downloaded"] += 1
        else:
            stats["errors"] += 1
    return stats


def get_topic_config_by_id(topic_id: str) -> Optional[TopicConfig]:
    """Возвращает конфиг топика по topic_id или None."""
    for t in TOPICS_CONFIG:
        if t.topic_id == topic_id:
            return t
    return None
