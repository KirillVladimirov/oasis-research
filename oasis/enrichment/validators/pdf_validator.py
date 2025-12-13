"""Валидация соответствия PDF статьям."""

import hashlib
import re
from typing import Any, Callable

import fitz  # PyMuPDF
import requests
from loguru import logger
from rapidfuzz import fuzz
from unidecode import unidecode

from oasis.enrichment.validators.cache import get_from_cache, put_in_cache
from oasis.enrichment.validators.title_matcher import normalize_title_for_match
from oasis.enrichment.validators.url_validator import is_trusted_source
from oasis.utils.text import clean_text_value, normalize_text, validate_doi_format

# Импортируем strict_title_match из matchers
try:
    from oasis.enrichment.cascade.matchers import strict_title_match
except ImportError:
    # Fallback если не удалось импортировать
    def strict_title_match(expected: str, candidate: str, threshold: float = 0.90) -> bool:
        """Fallback для strict_title_match."""
        if not expected or not candidate:
            return False
        ratio = fuzz.ratio(normalize_title_for_match(expected), normalize_title_for_match(candidate)) / 100.0
        return ratio >= threshold


def _safe_str(val: Any) -> str:
    """Безопасное преобразование в строку."""
    if val is None:
        return ""
    try:
        import pandas as _pd  # type: ignore

        if _pd.isna(val):
            return ""
    except Exception:
        pass
    try:
        return str(val).strip()
    except Exception:
        return ""


def _validate_arxiv_match(
    pdf_url: str,
    expected_arxiv: str,
    expected_title: str,
    session: requests.Session,
    arxiv_api_url: str,
) -> dict[str, Any] | None:
    """Быстрая валидация для arXiv PDF.

    Args:
        pdf_url: URL PDF файла
        expected_arxiv: Ожидаемый arXiv ID
        expected_title: Ожидаемый заголовок
        session: requests.Session
        arxiv_api_url: URL arXiv API

    Returns:
        Результат валидации или None если не совпало
    """
    if "arxiv.org" not in pdf_url.lower() or not expected_arxiv:
        return None

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,})", pdf_url.lower())
    if not arxiv_match:
        return None

    pdf_arxiv_id = arxiv_match.group(1)
    
    # Если ожидаемый arXiv ID указан, но не совпадает с PDF - сразу отклоняем
    if expected_arxiv and pdf_arxiv_id != expected_arxiv:
        return None
    
    # ВАЖНО: НЕ используем быструю ветку без проверки заголовка!
    # Даже если arXiv ID совпадает, нужно проверить заголовок через API,
    # чтобы избежать принятия неправильных статей с тем же arXiv ID
    
    # Получаем метаданные через arXiv API для проверки title
    try:
        r = session.get(
            arxiv_api_url,
            params={"id_list": pdf_arxiv_id, "max_results": 1},
            timeout=10,
        )
        if r.status_code == 200:
            import feedparser as _feedparser

            feed = _feedparser.parse(r.text)
            if feed.entries:
                entry = feed.entries[0]
                arxiv_title = clean_text_value(entry.get("title", ""))
                if arxiv_title and expected_title:
                    # Улучшенная нормализация для сравнения
                    norm_expected = normalize_title_for_match(expected_title)
                    norm_arxiv = normalize_title_for_match(arxiv_title)

                    # Проверка длины заголовков (если разница > 30%, понижаем confidence)
                    expected_len = len(norm_expected)
                    arxiv_len = len(norm_arxiv)
                    length_penalty = 0.0
                    if expected_len > 0 and arxiv_len > 0:
                        length_diff = abs(expected_len - arxiv_len) / max(
                            expected_len, arxiv_len
                        )
                        if length_diff > 0.3:
                            length_penalty = 0.15  # Штраф за большую разницу в длине

                    # Используем fuzz.ratio для длинных заголовков (>50 символов), иначе fuzz.partial_ratio
                    if expected_len > 50:
                        title_ratio = fuzz.ratio(norm_expected, norm_arxiv) / 100.0
                    else:
                        title_ratio = (
                            fuzz.partial_ratio(norm_expected, norm_arxiv) / 100.0
                        )

                    # Применяем штраф за длину
                    title_ratio = max(0.0, title_ratio - length_penalty)

                    # Для arXiv (доверенный источник) порог 0.75 - ниже чем для остальных
                    if title_ratio >= 0.75:
                        return {
                            "matched": True,
                            "confidence": 0.85 + (title_ratio - 0.75) * 0.15,  # 0.85-1.0 range
                            "extracted": {
                                "title": arxiv_title,
                                "arxiv_id": pdf_arxiv_id,
                                "source": "arxiv",
                            },
                        }
    except Exception:
        pass

    return None


def _validate_crossref_match(
    pdf_url: str,
    expected_doi: str,
    expected_title: str,
    get_crossref_fn: Callable[[str], dict[str, Any] | None] | None,
) -> dict[str, Any] | None:
    """Быстрая валидация для DOI через Crossref.

    Args:
        pdf_url: URL PDF файла
        expected_doi: Ожидаемый DOI
        expected_title: Ожидаемый заголовок
        get_crossref_fn: Функция для получения метаданных через Crossref

    Returns:
        Результат валидации или None если не совпало
    """
    if not expected_doi or not validate_doi_format(expected_doi) or not get_crossref_fn:
        return None

    try:
        cr = get_crossref_fn(expected_doi)
        if not cr:
            return None

        cr_title = clean_text_value((cr.get("title") or [None])[0] or "")
        if not cr_title or not expected_title:
            return None

        # Улучшенная нормализация для сравнения
        norm_expected = normalize_title_for_match(expected_title)
        norm_cr = normalize_title_for_match(cr_title)

        # Проверка длины заголовков
        expected_len = len(norm_expected)
        cr_len = len(norm_cr)
        length_penalty = 0.0
        if expected_len > 0 and cr_len > 0:
            length_diff = abs(expected_len - cr_len) / max(expected_len, cr_len)
            if length_diff > 0.3:
                length_penalty = 0.15

        # Используем fuzz.ratio для длинных заголовков
        if expected_len > 50:
            title_ratio = fuzz.ratio(norm_expected, norm_cr) / 100.0
        else:
            title_ratio = fuzz.partial_ratio(norm_expected, norm_cr) / 100.0

        title_ratio = max(0.0, title_ratio - length_penalty)

        # Для Crossref (доверенный источник) порог 0.85
        if title_ratio >= 0.85:
            # Проверяем что PDF URL содержит DOI или домен Crossref
            if expected_doi.lower() in pdf_url.lower() or any(
                d in pdf_url.lower() for d in ["doi.org", "crossref.org", "dx.doi.org"]
            ):
                return {
                    "matched": True,
                    "confidence": 0.9 + (title_ratio - 0.85) * 0.1,
                    "extracted": {
                        "title": cr_title,
                        "doi": expected_doi,
                        "source": "crossref",
                    },
                }
    except Exception:
        pass

    return None


def _extract_title_from_pdf_text(lines: list[str]) -> str:
    """Извлекает заголовок из текста PDF, объединяя строки с переносами.

    Args:
        lines: Строки текста PDF

    Returns:
        Извлечённый заголовок (объединенный из нескольких строк если нужно)
    """
    # Стратегия 1: Пробуем объединить соседние строки для заголовков с переносами
    # Ищем последовательность строк, которые могут быть частями заголовка
    candidate_sequences = []
    
    for i in range(len(lines)):
        line = lines[i].strip()
        # Пропускаем очень короткие строки (инициалы, номера страниц)
        if len(line) < 10:
            continue
        # Пропускаем очень длинные строки (обычно это абзацы)
        if len(line) > 250:
            continue
        # Пропускаем строки, которые выглядят как метаданные (авторы, даты)
        if any(
            keyword in line.lower()
            for keyword in [
                "abstract",
                "keywords",
                "introduction",
                "author",
                "received",
                "accepted",
            ]
        ):
            continue
        
        # Пробуем объединить с соседними строками (до 3 строк подряд)
        combined_title = line
        next_idx = i + 1
        combined_count = 1
        
        # Объединяем соседние строки, которые выглядят как продолжение заголовка
        while next_idx < len(lines) and combined_count < 3:
            next_line = lines[next_idx].strip()
            # Если следующая строка короткая и не содержит метаданных - возможно часть заголовка
            if (10 <= len(next_line) <= 150 and 
                not any(kw in next_line.lower() for kw in ["abstract", "keywords", "author", "introduction"])):
                # Объединяем с пробелом (убираем переносы строк)
                combined_title = f"{combined_title} {next_line}"
                combined_count += 1
                next_idx += 1
            else:
                break
        
        # Проверяем длину объединенного заголовка
        if 20 <= len(combined_title) <= 300:
            candidate_sequences.append((i, combined_title, combined_count))
    
    # Если нашли последовательности, предпочитаем те, что в начале (первые 15 строк)
    if candidate_sequences:
        # Сортируем по позиции (приоритет первым строкам)
        candidate_sequences.sort(key=lambda x: x[0])
        
        # Предпочитаем последовательности из первых 15 строк
        for idx, title, count in candidate_sequences:
            if idx < 15:
                # Нормализуем пробелы (убираем множественные)
                title = re.sub(r"\s+", " ", title).strip()
                return title
        
        # Если не нашли в первых 15, берем первый подходящий
        if candidate_sequences:
            title = candidate_sequences[0][1]
            title = re.sub(r"\s+", " ", title).strip()
            return title

    # Стратегия 2: Fallback - ищем одиночные строки (старый метод)
    candidate_lines = []
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if len(line_clean) < 15 or len(line_clean) > 250:
            continue
        if any(
            keyword in line_clean.lower()
            for keyword in ["abstract", "keywords", "introduction", "author", "received", "accepted"]
        ):
            continue
        candidate_lines.append((i, line_clean))

    if candidate_lines:
        for idx, line in candidate_lines[:10]:
            if idx < 10 and 20 <= len(line) <= 200:
                return line
        if candidate_lines:
            return candidate_lines[0][1]

    return ""


def _extract_metadata_from_pdf(
    pdf_content: bytes,
) -> tuple[str, str, str]:
    """Извлекает метаданные (title, DOI, arXiv ID) из PDF контента.

    Args:
        pdf_content: Байты PDF файла

    Returns:
        Кортеж (title, doi, arxiv_id)
    """
    extracted_title = ""
    extracted_doi = ""
    extracted_arxiv = ""

    try:
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        if pdf_doc.page_count > 0:
            # Берем текст первых 2 страниц БЕЗ нормализации (для сохранения структуры)
            first_pages_text_raw = ""
            for page_num in range(min(2, pdf_doc.page_count)):
                page = pdf_doc[page_num]
                first_pages_text_raw += page.get_text()

            # Извлекаем заголовок из исходного текста (сохраняем регистр и структуру)
            lines_raw = first_pages_text_raw.split("\n")[:30]
            extracted_title = _extract_title_from_pdf_text(lines_raw)
            
            # Нормализуем текст для поиска DOI и arXiv ID
            first_pages_text = normalize_text(unidecode(first_pages_text_raw.lower()))

            # Стратегия 2: Если не нашли, пробуем извлечь заголовок из первых абзацев
            if not extracted_title:
                # Ищем первую фразу длиной 20-200 символов в начале текста
                text_start = first_pages_text[:1000]  # Первые 1000 символов
                # Разбиваем на предложения (по точкам, восклицательным и вопросительным знакам)
                sentences = re.split(r"[.!?]\s+", text_start)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if 20 <= len(sentence_clean) <= 200:
                        # Проверяем, что это не метаданные
                        if not any(
                            keyword in sentence_clean.lower()
                            for keyword in [
                                "abstract",
                                "keywords",
                                "author",
                                "received",
                            ]
                        ):
                            extracted_title = sentence_clean
                            break

            # Поиск DOI в тексте
            doi_match = re.search(
                r"(?:doi[:\s]+|doi\.org/)(10\.\d{4,}/[^\s\)]+)",
                first_pages_text,
                re.IGNORECASE,
            )
            if doi_match:
                extracted_doi = doi_match.group(1).lower()

            # Поиск arXiv ID
            arxiv_match = re.search(
                r"arxiv[:\s]+(\d{4}\.\d{4,})", first_pages_text, re.IGNORECASE
            )
            if arxiv_match:
                extracted_arxiv = arxiv_match.group(1)

        pdf_doc.close()
    except Exception:
        pass

    return extracted_title, extracted_doi, extracted_arxiv


def validate_pdf_title(pdf_content: bytes, expected_title: str, threshold: float = 0.85) -> dict[str, Any]:
    """Строгая проверка заголовка в PDF документе.
    
    Извлекает заголовок из первых страниц PDF и проверяет соответствие
    ожидаемому названию статьи с использованием strict_title_match.
    
    Args:
        pdf_content: Байты PDF файла
        expected_title: Ожидаемое название статьи
        threshold: Минимальный порог similarity (по умолчанию 0.85)
        
    Returns:
        Словарь с результатом валидации:
        - matched: True если заголовок совпадает
        - confidence: Уверенность в совпадении (0.0-1.0)
        - extracted_title: Извлеченный заголовок (если есть)
    """
    if not expected_title:
        return {"matched": False, "confidence": 0.0}
    
    # Извлекаем заголовок из PDF
    extracted_title, _, _ = _extract_metadata_from_pdf(pdf_content)
    
    logger.debug(
        f"validate_pdf_title: ожидаемый='{expected_title[:60]}...', "
        f"извлеченный='{extracted_title[:60] if extracted_title else 'N/A'}...'"
    )
    
    if not extracted_title:
        # Не смогли извлечь заголовок - отклоняем
        logger.debug("validate_pdf_title: не удалось извлечь заголовок из PDF")
        return {
            "matched": False,
            "confidence": 0.0,
            "reason": "Не удалось извлечь заголовок из PDF"
        }
    
    # Используем строгую валидацию заголовка
    title_matches = strict_title_match(expected_title, extracted_title, threshold=threshold)
    logger.debug(
        f"validate_pdf_title: strict_title_match={title_matches} "
        f"(threshold={threshold:.3f})"
    )
    
    if not title_matches:
        # Заголовок не совпадает
        logger.debug(
            f"validate_pdf_title: заголовок НЕ совпадает - "
            f"извлеченный='{extracted_title[:80]}...' vs ожидаемый='{expected_title[:80]}...'"
        )
        return {
            "matched": False,
            "confidence": 0.0,
            "extracted_title": extracted_title,
            "reason": f"Заголовок PDF не совпадает: '{extracted_title[:60]}...' vs '{expected_title[:60]}...'"
        }
    
    # Заголовок совпадает - высокая уверенность
    logger.debug(
        f"validate_pdf_title: заголовок совпадает - "
        f"извлеченный='{extracted_title[:80]}...'"
    )
    return {
        "matched": True,
        "confidence": 0.95,
        "extracted_title": extracted_title
    }


def validate_pdf_title_by_url(
    pdf_url: str,
    expected_title: str,
    session: requests.Session | None = None,
    threshold: float = 0.85
) -> dict[str, Any]:
    """Загружает PDF по URL и проверяет заголовок.
    
    Args:
        pdf_url: URL PDF файла
        expected_title: Ожидаемое название статьи
        session: requests.Session для HTTP запросов (опционально)
        threshold: Минимальный порог similarity (по умолчанию 0.85)
        
    Returns:
        Словарь с результатом валидации
    """
    if not session:
        session = requests.Session()
    
    try:
        # Загружаем первые 2 МБ PDF
        r = session.get(
            pdf_url,
            headers={"Range": "bytes=0-2097152"},
            timeout=10,
            stream=True,
        )
        if r.status_code not in (200, 206):
            return {
                "matched": False,
                "confidence": 0.0,
                "reason": f"Не удалось загрузить PDF: HTTP {r.status_code}"
            }
        
        pdf_content = b""
        for chunk in r.iter_content(chunk_size=8192):
            pdf_content += chunk
            if len(pdf_content) >= 2097152:  # 2 МБ
                break
        
        # Валидируем заголовок
        return validate_pdf_title(pdf_content, expected_title, threshold=threshold)
        
    except Exception as e:
        return {
            "matched": False,
            "confidence": 0.0,
            "reason": f"Ошибка при загрузке/проверке PDF: {e}"
        }


def _validate_pdf_content_match(
    pdf_url: str,
    expected_title: str,
    expected_doi: str,
    expected_arxiv: str,
    session: requests.Session,
) -> dict[str, Any] | None:
    """Валидация через извлечение метаданных из PDF контента.

    Args:
        pdf_url: URL PDF файла
        expected_title: Ожидаемый заголовок
        expected_doi: Ожидаемый DOI
        expected_arxiv: Ожидаемый arXiv ID
        session: requests.Session

    Returns:
        Результат валидации или None если не совпало
    """
    try:
        # Запрос только первых 2 МБ PDF
        r = session.get(
            pdf_url,
            headers={"Range": "bytes=0-2097152"},
            timeout=10,
            stream=True,
        )
        if r.status_code not in (200, 206):  # 206 = Partial Content
            return None

        pdf_content = b""
        for chunk in r.iter_content(chunk_size=8192):
            pdf_content += chunk
            if len(pdf_content) >= 2097152:  # 2 МБ
                break

        # Попытка извлечь метаданные через PyMuPDF
        extracted_title, extracted_doi, extracted_arxiv = _extract_metadata_from_pdf(
            pdf_content
        )

        # Логируем извлеченные данные
        logger.debug(
            f"PDF валидация для {pdf_url[:60]}...: "
            f"извлечен_title='{extracted_title[:80] if extracted_title else 'N/A'}...', "
            f"извлечен_doi={extracted_doi or 'N/A'}, "
            f"извлечен_arxiv={extracted_arxiv or 'N/A'}"
        )

        # Сравнение извлеченных данных с ожидаемыми
        confidence = 0.0
        matched = False

        # Определяем, является ли источник доверенным
        is_trusted = is_trusted_source(pdf_url)
        # Порог для доверенных источников ниже (0.80), для остальных - 0.85
        title_threshold = 0.80 if is_trusted else 0.85

        # Проверка title с улучшенной нормализацией
        if extracted_title and expected_title:
            norm_extracted = normalize_title_for_match(extracted_title)
            norm_expected = normalize_title_for_match(expected_title)

            # Проверка длины заголовков
            expected_len = len(norm_expected)
            extracted_len = len(norm_extracted)
            length_penalty = 0.0
            if expected_len > 0 and extracted_len > 0:
                length_diff = abs(expected_len - extracted_len) / max(
                    expected_len, extracted_len
                )
                if length_diff > 0.3:
                    length_penalty = 0.15

            # Используем fuzz.ratio для длинных заголовков
            if expected_len > 50:
                title_ratio = fuzz.ratio(norm_expected, norm_extracted) / 100.0
            else:
                title_ratio = fuzz.partial_ratio(norm_expected, norm_extracted) / 100.0

            title_ratio = max(0.0, title_ratio - length_penalty)
            
            logger.debug(
                f"PDF валидация title: ожидаемый='{expected_title[:60]}...', "
                f"извлеченный='{extracted_title[:60]}...', "
                f"нормализованный_ожидаемый='{norm_expected[:60]}...', "
                f"нормализованный_извлеченный='{norm_extracted[:60]}...', "
                f"title_ratio={title_ratio:.3f}, "
                f"length_penalty={length_penalty:.3f}, "
                f"threshold={title_threshold:.3f}"
            )

            if title_ratio >= title_threshold:
                confidence = title_ratio
                matched = True
                logger.debug(f"PDF валидация: title совпадает (ratio={title_ratio:.3f} >= {title_threshold:.3f})")
            else:
                logger.debug(
                    f"PDF валидация: title НЕ совпадает (ratio={title_ratio:.3f} < {title_threshold:.3f})"
                )
        elif not extracted_title:
            logger.debug(f"PDF валидация: не удалось извлечь заголовок из PDF")
        elif not expected_title:
            logger.debug(f"PDF валидация: ожидаемый заголовок не указан")

        # Проверка DOI (повышает уверенность)
        if extracted_doi and expected_doi:
            if extracted_doi == expected_doi:
                confidence = min(1.0, confidence + 0.1)
                matched = True
            elif extracted_doi != expected_doi:
                # Разные DOI - явное несоответствие
                matched = False
                confidence = 0.0

        # Проверка arXiv ID (повышает уверенность)
        if extracted_arxiv and expected_arxiv:
            if extracted_arxiv == expected_arxiv:
                confidence = min(1.0, confidence + 0.1)
                matched = True
            elif extracted_arxiv != expected_arxiv:
                # Разные arXiv ID - явное несоответствие
                matched = False
                confidence = 0.0

        # Используем тот же порог для финальной проверки
        confidence_threshold = 0.80 if is_trusted else 0.85
        if matched and confidence >= confidence_threshold:
            logger.debug(
                f"PDF валидация: УСПЕХ (matched=True, confidence={confidence:.3f} >= {confidence_threshold:.3f})"
            )
            return {
                "matched": True,
                "confidence": confidence,
                "extracted": {
                    "title": extracted_title,
                    "doi": extracted_doi,
                    "arxiv_id": extracted_arxiv,
                    "source": "pdf_extraction",
                },
            }
        else:
            logger.debug(
                f"PDF валидация: ОТКЛОНЕНО (matched={matched}, confidence={confidence:.3f} < {confidence_threshold:.3f})"
            )

    except Exception:
        pass

    return None


def validate_pdf_matches_reference(
    pdf_url: str,
    expected: dict[str, Any],
    session: requests.Session | None = None,
    source_hint: str | None = None,
    get_crossref_fn: Callable[[str], dict[str, Any] | None] | None = None,
    arxiv_api_url: str = "https://export.arxiv.org/api/query",
) -> dict[str, Any]:
    """Проверяет что PDF соответствует искомой статье по title/DOI/arXiv ID.

    Args:
        session: requests.Session для выполнения запросов
        pdf_url: URL PDF файла
        expected: Словарь с ожидаемыми данными: {title, authors, year, doi, arxiv_id}
        source_hint: Подсказка об источнике ("arxiv", "crossref", etc.)
        get_crossref_fn: Опциональная функция для получения метаданных через Crossref по DOI
        arxiv_api_url: URL arXiv API (по умолчанию стандартный)

    Returns:
        dict: {
            "matched": bool,
            "confidence": float (0.0-1.0),
            "extracted": {title, authors, year, doi, arxiv_id, source}
        }
    """
    if not pdf_url or not isinstance(pdf_url, str):
        return {"matched": False, "confidence": 0.0, "extracted": {}}
    
    # Создаём сессию если не передана
    if session is None:
        session = requests.Session()

    expected_title = _safe_str(expected.get("title"))
    expected_doi = _safe_str(expected.get("doi")).lower()
    expected_arxiv = _safe_str(expected.get("arxiv_id"))

    # Кэш-ключ: pdf_url + hash(title)
    cache_key = (
        f"{pdf_url}#{hashlib.sha256(expected_title.encode('utf-8')).hexdigest()[:16]}"
    )
    cached = get_from_cache(cache_key)
    if cached:
        return cached

    # Быстрая ветка для arXiv
    result = _validate_arxiv_match(
        pdf_url, expected_arxiv, expected_title, session, arxiv_api_url
    )
    if result:
        put_in_cache(cache_key, result)
        return result

    # Быстрая ветка для DOI (проверка через Crossref)
    result = _validate_crossref_match(
        pdf_url, expected_doi, expected_title, get_crossref_fn
    )
    if result:
        put_in_cache(cache_key, result)
        return result

    # Fallback: извлечение метаданных из PDF (первые 1-2 МБ)
    result = _validate_pdf_content_match(
        pdf_url, expected_title, expected_doi, expected_arxiv, session
    )
    if result:
        put_in_cache(cache_key, result)
        return result

    # По умолчанию - не совпадает
    result = {"matched": False, "confidence": 0.0, "extracted": {}}
    put_in_cache(cache_key, result)
    return result
