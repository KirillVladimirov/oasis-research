"""Скрипт валидации PDF из CSV файла.

Скачивает все PDF из ссылок в CSV, проверяет заголовки, сохраняет правильные и удаляет неправильные.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import hashlib
import threading
import queue

import pandas as pd
import requests
from requests.exceptions import Timeout as RequestsTimeout
from loguru import logger
from tqdm import tqdm
import fitz  # PyMuPDF

# Загружаем переменные окружения из .env ДО настройки логирования
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не обязателен, но желателен

from oasis.enrichment.validators.pdf_validator import _extract_metadata_from_pdf

# Настройка логирования
logger.remove()
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"pdf_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")

# Черный список доменов, которые часто зависают - пропускаем HEAD запросы
PROBLEMATIC_DOMAINS = {
    "semanticscholar.org",
    "www.semanticscholar.org",
    "scholar.google.com",
    "www.google.com",
    "yandex.ru",
    "www.yandex.ru",
}


def _head_request_with_timeout(url: str, headers: dict, timeout: float) -> tuple[bool, str]:
    """Выполняет HEAD запрос с принудительным таймаутом через поток."""
    logger.debug(f"[_head_request_with_timeout] Запрос для {url}, timeout={timeout}")
    result_queue = queue.Queue()
    
    def _do_request():
        try:
            logger.debug(f"[_head_request_with_timeout] Поток начал выполнение для {url}")
            temp_session = requests.Session()
            temp_session.headers.update(headers)
            try:
                logger.debug(f"[_head_request_with_timeout] Выполняем HEAD запрос для {url}")
                response = temp_session.head(url, timeout=timeout, allow_redirects=True)
                logger.debug(f"[_head_request_with_timeout] HEAD запрос завершен для {url}, status={response.status_code}")
                content_type = response.headers.get("Content-Type", "").lower()
                
                if "application/pdf" in content_type or "pdf" in content_type:
                    result_queue.put((True, "pdf"))
                elif response.status_code == 200:
                    result_queue.put((True, "website"))
                else:
                    result_queue.put((True, "error"))
            finally:
                temp_session.close()
                logger.debug(f"[_head_request_with_timeout] Сессия закрыта для {url}")
        except Exception as e:
            logger.error(f"[_head_request_with_timeout] Ошибка в потоке для {url}: {e}")
            result_queue.put((False, str(e)))
    
    thread = threading.Thread(target=_do_request, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 0.5)
    
    if thread.is_alive():
        logger.warning(f"[_head_request_with_timeout] Таймаут при HEAD запросе к {url}")
        return (False, "timeout")
    
    try:
        success, result = result_queue.get(timeout=0.1)
        return (success, result)
    except queue.Empty:
        logger.warning(f"[_head_request_with_timeout] Очередь пуста для {url}")
        return (False, "timeout")


def detect_resource_type(url: str, session: requests.Session) -> str:
    """Определяет тип ресурса по URL и заголовкам."""
    try:
        url_lower = url.lower()
        if url_lower.endswith(".pdf") or "/pdf" in url_lower.split("?")[0]:
            logger.debug(f"[detect_resource_type] Определено как PDF по расширению: {url}")
            return "pdf"
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        if domain in PROBLEMATIC_DOMAINS:
            return "website"
        
        success, result = _head_request_with_timeout(url, session.headers, timeout=3.0)
        
        if not success:
            logger.debug(f"[detect_resource_type] Ошибка при определении типа ресурса {url}: {result}")
            return "error"
        
        return result
    except Exception as e:
        logger.error(f"[detect_resource_type] Исключение при определении типа ресурса {url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "error"


def download_pdf(url: str, session: requests.Session, temp_dir: Path) -> Path | None:
    """Скачивает PDF в временную директорию.
    
    Args:
        url: URL PDF файла
        session: requests.Session для HTTP запросов
        temp_dir: Временная директория для сохранения
        
    Returns:
        Путь к скачанному файлу или None при ошибке
    """
    try:
        # Загружаем первые 2 МБ PDF (как в validate_pdf_title_by_url) с таймаутом
        response = session.get(
            url,
            headers={"Range": "bytes=0-2097152"},
            timeout=15,
            stream=True,
        )
        
        if response.status_code not in (200, 206):
            logger.debug(f"Не удалось загрузить PDF {url}: HTTP {response.status_code}")
            return None
        
        # Создаем имя файла на основе URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        pdf_path = temp_dir / f"{url_hash}.pdf"
        
        # Сохраняем PDF
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                if pdf_path.stat().st_size >= 2097152:  # 2 МБ
                    break
        
        logger.debug(f"Скачан PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)")
        return pdf_path
        
    except RequestsTimeout:
        logger.debug(f"Таймаут при скачивании PDF {url}")
        return None
    except Exception as e:
        logger.debug(f"Ошибка при скачивании PDF {url}: {e}")
        return None


def extract_pdf_title(pdf_path: Path) -> str:
    """Извлекает заголовок из PDF файла с детальным логированием.
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Извлеченный заголовок или пустая строка
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        # Детальное логирование: извлекаем больше текста для лучшего извлечения заголовка
        import fitz
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        lines_raw = []
        if pdf_doc.page_count > 0:
            first_pages_text_raw = ""
            # Берем больше страниц (до 3) для лучшего извлечения заголовка
            for page_num in range(min(3, pdf_doc.page_count)):
                page = pdf_doc[page_num]
                first_pages_text_raw += page.get_text()
            
            # Берем больше строк (до 100) для лучшего поиска заголовка
            lines_raw = first_pages_text_raw.split("\n")[:100]
            logger.debug(f"Первые 100 строк PDF {pdf_path.name}:")
            for idx, line in enumerate(lines_raw[:50]):  # Логируем первые 50 для краткости
                logger.debug(f"  [{idx:2d}] {line}")
        
        pdf_doc.close()
        
        # Список шаблонов для фильтрации
        template_patterns = [
            r"journal of latex class files",
            r"ieee transactions?",
            r"acm",
            r"vol\.?\s*\d+",
            r"no\.?\s*\d+",
            r"august|september|october|november|december|january|february|march|april|may|june|july",
            r"manuscript received",
            r"final manuscript",
            r"preprint",
            r"submitted to",
            r"draft",
        ]
        
        # Сначала пробуем использовать _extract_metadata_from_pdf, но улучшим поиск
        extracted_title, _, _ = _extract_metadata_from_pdf(pdf_content)
        logger.debug(f"Извлеченный заголовок (из _extract_metadata_from_pdf): '{extracted_title}'")
        
        # Очищаем заголовок от авторов и метаданных
        if extracted_title:
            # Убираем авторов в конце заголовка
            # Паттерны авторов:
            # 1. "Имя Фамилия" - два слова с заглавной буквы подряд
            # 2. "Имя Фамилия Число" - имя, фамилия, число
            # 3. "Имя Фамилия Имя2 Фамилия2" - несколько имен подряд
            title_words = extracted_title.split()
            cleaned_words = []
            i = 0
            
            # Ищем начало авторов - место, где начинается последовательность имен
            author_start_idx = None
            for idx in range(len(title_words)):
                word = title_words[idx]
                # Если слово - имя (заглавная буква, короткое)
                if re.match(r'^[A-Z][a-z]+$', word) and len(word) < 20:
                    # Проверяем следующие слова для определения паттерна авторов
                    if idx + 1 < len(title_words):
                        next_word = title_words[idx + 1]
                        # Паттерн 1: "Имя Фамилия" - следующее слово тоже имя
                        if re.match(r'^[A-Z][a-z]+$', next_word) and len(next_word) < 20:
                            # Проверяем, что это не часть заголовка
                            # Если после этого идет число или еще одно имя - это авторы
                            if idx + 2 < len(title_words):
                                next_next = title_words[idx + 2]
                                if re.match(r'^\d+$', next_next) or re.match(r'^[A-Z][a-z]+$', next_next):
                                    author_start_idx = idx
                                    logger.debug(f"Обнаружены авторы в заголовке на позиции {idx}: '{word} {next_word}...'")
                                    break
                            # Если это последние слова и они выглядят как имена - тоже авторы
                            elif idx >= len(title_words) - 3:  # Последние 3 слова
                                author_start_idx = idx
                                logger.debug(f"Обнаружены авторы в конце заголовка на позиции {idx}: '{word} {next_word}...'")
                                break
                        # Паттерн 2: "Имя Фамилия Число"
                        elif re.match(r'^\d+$', next_word) and idx >= len(title_words) - 3:
                            author_start_idx = idx
                            logger.debug(f"Обнаружены авторы с номером на позиции {idx}: '{word} {next_word}...'")
                            break
            
            # Если нашли начало авторов - обрезаем заголовок
            if author_start_idx is not None:
                extracted_title = " ".join(title_words[:author_start_idx]).strip()
                logger.debug(f"Извлеченный заголовок (после очистки от авторов): '{extracted_title}'")
            else:
                logger.debug(f"Авторы не обнаружены в заголовке")
        
        # Фильтруем шаблоны LaTeX и шаблонные строки
        if extracted_title:
            title_lower = extracted_title.lower()
            for pattern in template_patterns:
                if re.search(pattern, title_lower):
                    logger.debug(f"Отфильтрован шаблонный заголовок: '{extracted_title}' (паттерн: {pattern})")
                    # Пробуем найти следующий кандидат
                    extracted_title = ""
                    break
        
        # Проверяем, является ли извлеченный заголовок авторами
        def is_authors_line(line: str) -> bool:
            """Проверяет, является ли строка списком авторов."""
            line_clean = line.strip()
            if not line_clean:
                return False
            
            line_lower = line_clean.lower()
            
            # Признаки авторов:
            # 1. Содержит "Member, IEEE", "Fellow, IEEE", "ACM"
            if any(marker in line_lower for marker in ["member,", "fellow,", "ieee", "acm"]):
                return True
            
            # 2. Много запятых (список авторов через запятую)
            if line_clean.count(",") >= 2:
                return True
            
            # 3. Паттерн "Имя Фамилия, Имя Фамилия" - несколько имен подряд с запятыми
            words = line_clean.split()
            if len(words) >= 3:
                # Проверяем, есть ли несколько пар "Имя Фамилия" подряд
                name_pairs = 0
                for i in range(len(words) - 1):
                    word = words[i].rstrip(",")
                    next_word = words[i + 1].rstrip(",")
                    if (re.match(r'^[A-Z][a-z]+$', word) and len(word) < 20 and
                        re.match(r'^[A-Z][a-z]+$', next_word) and len(next_word) < 20):
                        name_pairs += 1
                if name_pairs >= 2:  # Если есть 2+ пары имен - это авторы
                    return True
            
            # 4. Короткая строка с именами (менее 50 символов и содержит имена)
            if len(line_clean) < 50:
                words = line_clean.split()
                if len(words) >= 2:
                    # Проверяем первые два слова - если это имена, то это авторы
                    first_word = words[0].rstrip(",")
                    second_word = words[1].rstrip(",")
                    if (re.match(r'^[A-Z][a-z]+$', first_word) and len(first_word) < 20 and
                        re.match(r'^[A-Z][a-z]+$', second_word) and len(second_word) < 20):
                        return True
            
            return False
        
        # Проверяем, является ли извлеченный заголовок авторами
        title_looks_like_authors = False
        if extracted_title:
            if is_authors_line(extracted_title):
                title_looks_like_authors = True
                logger.debug(f"Найденный заголовок является авторами: '{extracted_title}'")
        
        # Если заголовок не найден или является авторами, ищем в raw тексте
        if (not extracted_title or title_looks_like_authors) and lines_raw:
            logger.debug("Пробуем найти альтернативный заголовок из raw текста...")
            # Ищем строки, которые не являются шаблонами, и объединяем соседние
            # Ищем в первых 50 строках для лучшего покрытия
            for idx in range(min(50, len(lines_raw))):
                line_clean = lines_raw[idx].strip()
                if len(line_clean) < 15:  # Пропускаем очень короткие строки (минимум 15 символов)
                    continue
                
                # Проверяем, что это не шаблон
                line_lower = line_clean.lower()
                is_template = any(
                    re.search(pattern, line_lower) 
                    for pattern in template_patterns
                ) or any(
                    keyword in line_lower
                    for keyword in ["abstract", "keywords", "introduction", "author", "received", "accepted"]
                )
                
                # Пропускаем строки, которые являются авторами
                if is_authors_line(line_clean):
                    logger.debug(f"Пропускаем строку {idx} (авторы): '{line_clean[:60]}...'")
                    continue
                
                if not is_template:
                    # Пробуем объединить с соседними строками (до 5 строк для длинных заголовков)
                    combined_title = line_clean
                    next_idx = idx + 1
                    combined_count = 1
                    
                    while next_idx < len(lines_raw) and combined_count < 5 and next_idx < 50:
                        next_line = lines_raw[next_idx].strip()
                        next_line_lower = next_line.lower()
                        
                        # Проверяем, является ли следующая строка авторами
                        if is_authors_line(next_line):
                            logger.debug(f"Останавливаемся на строке {next_idx} (авторы): '{next_line[:60]}...'")
                            break
                        
                        # Признаки метаданных (не авторов) - останавливаемся
                        is_metadata_line = (
                            # Слишком длинная строка (обычно это абзац текста)
                            len(next_line) > 150 or
                            # Слишком короткая строка
                            len(next_line) < 10
                        )
                        
                        # Проверяем, что это не шаблон
                        is_template_next = any(
                            re.search(pattern, next_line_lower) 
                            for pattern in template_patterns
                        ) or any(
                            keyword in next_line_lower
                            for keyword in ["abstract", "keywords", "introduction", "author", "received", "accepted"]
                        )
                        
                        # Если следующая строка подходит для заголовка
                        if (len(next_line) >= 10 and len(next_line) <= 150 and 
                            not is_metadata_line and
                            not is_template_next and
                            not is_authors_line(next_line)):
                            # Объединяем с пробелом
                            combined_title = f"{combined_title} {next_line}"
                            combined_count += 1
                            next_idx += 1
                        else:
                            # Останавливаемся - встретили метаданные или шаблоны
                            break
                    
                    # Очищаем заголовок от авторов и метаданных в конце
                    # Убираем строки, которые выглядят как имена авторов
                    title_words = combined_title.split()
                    cleaned_words = []
                    for word in title_words:
                        # Если слово выглядит как имя (начинается с заглавной и короткое) или содержит только цифры
                        if re.match(r'^[A-Z][a-z]+$', word) and len(word) < 15:
                            # Проверяем, не является ли это частью заголовка
                            # Если после этого слова идет еще одно похожее слово - возможно авторы
                            word_idx = title_words.index(word)
                            if word_idx + 1 < len(title_words):
                                next_word = title_words[word_idx + 1]
                                # Если следующее слово тоже похоже на имя - это авторы
                                if re.match(r'^[A-Z][a-z]+(?:\d+)?$', next_word):
                                    # Останавливаемся здесь
                                    break
                        # Если слово заканчивается на цифру - возможно номер автора
                        if re.match(r'^\d+$', word) and len(cleaned_words) > 0:
                            # Проверяем предыдущее слово
                            prev_word = cleaned_words[-1] if cleaned_words else ""
                            if re.match(r'^[A-Z][a-z]+$', prev_word):
                                # Это номер автора - удаляем его и предыдущее слово
                                if cleaned_words:
                                    cleaned_words.pop()
                                break
                        cleaned_words.append(word)
                    
                    combined_title = " ".join(cleaned_words).strip()
                    
                    # Проверяем длину объединенного заголовка
                    if 20 <= len(combined_title) <= 300:
                        extracted_title = combined_title
                        logger.debug(f"Найден альтернативный заголовок (объединено {combined_count} строк, начиная с {idx}): '{extracted_title}'")
                        break
        
        return extracted_title or ""
    except Exception as e:
        logger.debug(f"Ошибка при извлечении заголовка из {pdf_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""


def validate_pdf_with_llm(
    expected_title: str,
    extracted_title: str,
    year: Any,
    authors: Any,
    venue: Any,
) -> dict[str, Any]:
    """Валидирует PDF через LLM (DeepSeek), если строгая валидация не прошла.
    
    Args:
        expected_title: Ожидаемый заголовок из CSV
        extracted_title: Извлеченный заголовок из PDF
        year: Год публикации
        authors: Авторы статьи
        venue: Издание/конференция
        
    Returns:
        Словарь с результатом: matched, confidence, reason, llm_response
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.deepseek.com/v1"
    
    if not api_key:
        logger.warning("API ключ не найден (DEEPSEEK_API_KEY или OPENAI_API_KEY), пропускаем LLM валидацию")
        return {
            "matched": False,
            "confidence": 0.0,
            "reason": "API ключ не найден",
            "llm_used": False
        }
    
    # Форматируем данные для промпта
    year_str = str(year) if year and not pd.isna(year) else "не указан"
    
    # Парсим авторов
    authors_list = []
    if authors:
        if isinstance(authors, str):
            # Пробуем распарсить как JSON или Python список
            try:
                import ast
                authors_list = ast.literal_eval(authors)
            except:
                # Разделяем по разделителям
                for sep in [";", "|", ","]:
                    if sep in authors:
                        authors_list = [a.strip() for a in authors.split(sep) if a.strip()]
                        break
                if not authors_list:
                    authors_list = [authors]
        elif isinstance(authors, list):
            authors_list = [str(a) for a in authors if a]
    
    authors_str = ", ".join(authors_list) if authors_list else "не указаны"
    venue_str = str(venue) if venue and not pd.isna(venue) else "не указано"
    
    # Строим промпт
    system_prompt = """Ты - эксперт по валидации научных статей. Твоя задача - определить, соответствует ли извлеченный из PDF заголовок ожидаемому заголовку статьи.

Учитывай:
1. Заголовки могут отличаться регистром, пунктуацией, порядком слов
2. В извлеченном заголовке могут отсутствовать авторы, год, издание
3. Важно семантическое совпадение основных терминов и смысла

Отвечай строго в формате JSON:
{
  "matched": true/false,
  "confidence": 0.0-1.0,
  "reason": "краткое объяснение на русском"
}"""
    
    user_prompt = f"""Ожидаемый заголовок статьи:
{expected_title}

Извлеченный заголовок из PDF:
{extracted_title}

Дополнительная информация:
- Год: {year_str}
- Авторы: {authors_str}
- Издание/конференция: {venue_str}

Определи, соответствует ли извлеченный заголовок ожидаемому. Учитывай, что в извлеченном заголовке могут быть пропущены авторы и другие метаданные, но основные термины и смысл должны совпадать."""
    
    try:
        logger.debug(f"[LLM валидация] Отправляем запрос к DeepSeek для: '{expected_title}' vs '{extracted_title}'")
        
        # Делаем запрос к DeepSeek API
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        llm_result = json.loads(content)
        
        logger.debug(f"[LLM валидация] Результат: {llm_result}")
        
        return {
            "matched": llm_result.get("matched", False),
            "confidence": float(llm_result.get("confidence", 0.0)),
            "reason": llm_result.get("reason", "LLM валидация"),
            "extracted_title": extracted_title,
            "llm_used": True,
            "llm_response": llm_result
        }
        
    except Exception as e:
        logger.error(f"[LLM валидация] Ошибка при запросе к DeepSeek: {e}")
        return {
            "matched": False,
            "confidence": 0.0,
            "reason": f"Ошибка LLM валидации: {e}",
            "llm_used": False
        }


def validate_pdf_title_match(
    expected_title: str, 
    extracted_title: str,
    year: Any = None,
    authors: Any = None,
    venue: Any = None,
) -> dict[str, Any]:
    """Валидирует совпадение заголовка PDF с ожидаемым через LLM.
    
    Args:
        expected_title: Ожидаемый заголовок из CSV
        extracted_title: Извлеченный заголовок из PDF
        year: Год публикации
        authors: Авторы статьи
        venue: Издание/конференция
        
    Returns:
        Словарь с результатом: matched, confidence, reason
    """
    if not expected_title or not extracted_title:
        return {
            "matched": False,
            "confidence": 0.0,
            "reason": "Отсутствует заголовок",
            "llm_used": False
        }
    
    # Используем LLM для валидации
    logger.info(f"[Валидация] Используем LLM валидацию для: '{expected_title}' vs '{extracted_title}'")
    llm_result = validate_pdf_with_llm(expected_title, extracted_title, year, authors, venue)
    return llm_result


def parse_additional_urls(val: Any) -> list[str]:
    """Парсит additional_urls из различных форматов.
    
    Args:
        val: Может быть list, JSON строка, строка с разделителями или пустое значение
        
    Returns:
        Список URL
    """
    if val is None:
        return []
    
    # Если уже список
    if isinstance(val, list):
        return [str(url).strip() for url in val if url]
    
    # Проверяем NaN
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in {"nan", "none", "null", "[]", ""}:
        return []
    
    # Пробуем распарсить как JSON
    if val_str.startswith("[") or val_str.startswith("{"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(url).strip() for url in parsed if url]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Пробуем разделить по разделителям
    for sep in ["|", ";", ",", "\n"]:
        if sep in val_str:
            urls = [url.strip() for url in val_str.split(sep) if url.strip()]
            if urls:
                return urls
    
    # Если ничего не подошло - возвращаем как одну строку (если похоже на URL)
    if val_str.startswith(("http://", "https://")):
        return [val_str]
    
    return []


def get_filename_from_url(url: str) -> str:
    """Извлекает имя файла из URL.
    
    Args:
        url: URL для парсинга
        
    Returns:
        Имя файла или "document.pdf" если не удалось извлечь
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        if path:
            filename = Path(path).name
            if filename and filename.endswith(".pdf"):
                return filename
        # Если не нашли в пути, пробуем из query параметров
        if parsed.query:
            match = re.search(r"filename=([^&]+\.pdf)", parsed.query, re.IGNORECASE)
            if match:
                return match.group(1)
        return "document.pdf"
    except Exception:
        return "document.pdf"


def process_reference(
    row: pd.Series,
    session: requests.Session,
    temp_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Обрабатывает одну статью: собирает ссылки, проверяет PDF, сохраняет правильные.
    
    Args:
        row: Строка из CSV (pd.Series)
        session: requests.Session для HTTP запросов
        temp_dir: Временная директория для скачивания
        output_dir: Директория для сохранения правильных PDF
        
    Returns:
        Словарь с результатами обработки
    """
    ref_number = str(row.get("ref_number", "")).strip()
    title = str(row.get("title", "")).strip()
    
    result = {
        "ref_number": ref_number,
        "title": title,
        "urls": [],
        "correct_pdfs": [],
        "incorrect_pdfs": [],
        "websites": [],
        "errors": [],
    }
    
    # Собираем все ссылки
    urls_to_check = []
    
    # pdf_url
    pdf_url = str(row.get("pdf_url", "")).strip()
    if pdf_url and pdf_url.lower() not in {"nan", "none", ""}:
        urls_to_check.append(("pdf_url", pdf_url))
    
    # article_url
    article_url = str(row.get("article_url", "")).strip()
    if article_url and article_url.lower() not in {"nan", "none", ""}:
        urls_to_check.append(("article_url", article_url))
    
    # additional_urls
    additional_urls = parse_additional_urls(row.get("additional_urls", ""))
    for idx, url in enumerate(additional_urls):
        urls_to_check.append((f"additional_urls[{idx}]", url))
    
    logger.info(f"Обработка статьи {ref_number}: {title} ({len(urls_to_check)} ссылок)")
    
    # Обрабатываем каждую ссылку
    for url_source, url in urls_to_check:
        logger.info(f"  [ШАГ 1] Начинаем обработку {url_source}: {url}")
        url_result = {
            "source": url_source,
            "url": url,
            "type": None,
            "status": None,
            "extracted_title": None,
            "reason": None,
            "saved_path": None,
        }
        
        try:
            # Определяем тип ресурса с обработкой зависаний
            logger.info(f"  [ШАГ 2] Определяем тип ресурса для {url_source}: {url}")
            try:
                resource_type = detect_resource_type(url, session)
                logger.info(f"  [ШАГ 2] Результат определения типа: {resource_type}")
            except Exception as detect_error:
                logger.error(f"  [ШАГ 2] Критическая ошибка при detect_resource_type для {url}: {detect_error}")
                resource_type = "error"
            
            url_result["type"] = resource_type
            logger.info(f"  [ШАГ 3] Тип ресурса установлен: {resource_type}")
            
            if resource_type == "pdf":
                # Скачиваем PDF
                logger.info(f"  [ШАГ 4] Начинаем скачивание PDF для {url_source}: {url}")
                pdf_path = download_pdf(url, session, temp_dir)
                logger.info(f"  [ШАГ 4] Результат скачивания: {pdf_path}")
                if not pdf_path:
                    url_result["status"] = "error"
                    url_result["reason"] = "Не удалось скачать PDF"
                    result["errors"].append(url_result)
                    continue
                
                # Извлекаем заголовок
                logger.info(f"  [ШАГ 5] Извлекаем заголовок из PDF: {pdf_path.name}")
                extracted_title = extract_pdf_title(pdf_path)
                logger.info(f"  [ШАГ 5] Извлеченный заголовок: '{extracted_title}'")
                url_result["extracted_title"] = extracted_title
                
                if not extracted_title:
                    # Не удалось извлечь заголовок - удаляем
                    pdf_path.unlink()
                    url_result["status"] = "error"
                    url_result["reason"] = "Не удалось извлечь заголовок из PDF"
                    result["errors"].append(url_result)
                    continue
                
                # Валидируем заголовок
                logger.info(f"  [ШАГ 6] Валидируем заголовок: ожидаемый='{title}', извлеченный='{extracted_title}'")
                # Получаем дополнительные данные для LLM валидации
                year = row.get("year")
                authors = row.get("authors")
                venue = row.get("venue")
                validation = validate_pdf_title_match(title, extracted_title, year, authors, venue)
                logger.info(f"  [ШАГ 6] Результат валидации: matched={validation['matched']}, confidence={validation['confidence']}, llm_used={validation.get('llm_used', False)}")
                url_result["reason"] = validation.get("reason", "")
                
                if validation["matched"]:
                    # Правильный PDF - сохраняем
                    filename = get_filename_from_url(url)
                    output_filename = f"{ref_number}_{filename}"
                    output_path = output_dir / output_filename
                    
                    # Если файл уже существует, добавляем суффикс
                    if output_path.exists():
                        base_name = output_filename.rsplit(".pdf", 1)[0]
                        counter = 1
                        while output_path.exists():
                            output_filename = f"{base_name}_{counter}.pdf"
                            output_path = output_dir / output_filename
                            counter += 1
                    
                    # Копируем файл
                    import shutil
                    shutil.copy2(pdf_path, output_path)
                    pdf_path.unlink()  # Удаляем временный файл
                    
                    url_result["status"] = "correct"
                    url_result["saved_path"] = str(output_path)
                    result["correct_pdfs"].append(url_result)
                    logger.info(f"   {url_source}: {url} → сохранен как {output_filename}")
                else:
                    # Неправильный PDF - удаляем
                    pdf_path.unlink()
                    url_result["status"] = "incorrect"
                    result["incorrect_pdfs"].append(url_result)
                    logger.info(f"   {url_source}: {url} → неправильный PDF")
                    
            elif resource_type == "website":
                url_result["status"] = "website"
                result["websites"].append(url_result)
                logger.info(f"   {url_source}: {url} → сайт")
                
            else:  # error
                url_result["status"] = "error"
                url_result["reason"] = "Не удалось определить тип ресурса"
                result["errors"].append(url_result)
                logger.info(f"  ️ {url_source}: {url} → ошибка")
                # Сразу переходим к следующей ссылке
                result["urls"].append(url_result)
                continue
                
        except Exception as e:
            url_result["status"] = "error"
            url_result["reason"] = f"Ошибка обработки: {e}"
            result["errors"].append(url_result)
            logger.error(f"   Ошибка при обработке {url_source}: {e}")
            # Сразу переходим к следующей ссылке
            result["urls"].append(url_result)
            continue
        
        # Добавляем результат только если не было continue
        result["urls"].append(url_result)
    
    return result


def generate_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Генерирует Markdown отчет с результатами валидации.
    
    Args:
        results: Список результатов обработки статей
        output_path: Путь для сохранения отчета
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Отчет валидации PDF\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Статистика
        total_refs = len(results)
        total_urls = sum(len(r["urls"]) for r in results)
        total_pdfs = sum(len(r["correct_pdfs"]) + len(r["incorrect_pdfs"]) for r in results)
        total_correct = sum(len(r["correct_pdfs"]) for r in results)
        total_incorrect = sum(len(r["incorrect_pdfs"]) for r in results)
        total_websites = sum(len(r["websites"]) for r in results)
        total_errors = sum(len(r["errors"]) for r in results)
        
        # Подсчет статей с несколькими правильными PDF
        refs_with_multiple_correct = sum(1 for r in results if len(r["correct_pdfs"]) > 1)
        
        f.write("## Статистика\n\n")
        f.write(f"- **Всего статей:** {total_refs}\n")
        f.write(f"- **Всего ссылок:** {total_urls}\n")
        f.write(f"- **PDF найдено:** {total_pdfs}\n")
        f.write(f"- **PDF правильных:** {total_correct}\n")
        f.write(f"- **PDF неправильных:** {total_incorrect}\n")
        f.write(f"- **Сайтов:** {total_websites}\n")
        f.write(f"- **Ошибок:** {total_errors}\n")
        if refs_with_multiple_correct > 0:
            f.write(f"- **️ Статей с несколькими правильными PDF (требуют ручного разрешения):** {refs_with_multiple_correct}\n")
        f.write("\n---\n\n")
        
        # Детали по каждой статье
        for result in results:
            ref_num = result["ref_number"]
            title = result["title"]
            
            f.write(f"## Статья {ref_num}: {title}\n\n")
            f.write(f"**Ref Number:** {ref_num}\n\n")
            
            if not result["urls"]:
                f.write("*Нет ссылок для проверки*\n\n")
                continue
            
            f.write("### Ссылки:\n\n")
            
            for url_result in result["urls"]:
                source = url_result["source"]
                url = url_result["url"]
                status = url_result["status"]
                reason = url_result.get("reason", "")
                extracted_title = url_result.get("extracted_title", "")
                
                if status == "correct":
                    emoji = ""
                    status_text = "PDF правильный"
                    if extracted_title:
                        status_text += f" (заголовок: \"{extracted_title}\")"
                    saved_path = url_result.get("saved_path", "")
                    if saved_path:
                        status_text += f" → сохранен: `{Path(saved_path).name}`"
                elif status == "incorrect":
                    emoji = ""
                    status_text = "PDF неправильный"
                    if extracted_title:
                        status_text += f" (заголовок: \"{extracted_title}\")"
                    if reason:
                        status_text += f" ({reason})"
                elif status == "website":
                    emoji = ""
                    status_text = "Сайт"
                else:  # error
                    emoji = "️"
                    status_text = "Ошибка"
                    if reason:
                        status_text += f": {reason}"
                
                f.write(f"- `{source}`: [{url}]({url}) → {emoji} {status_text}\n")
            
            # Если несколько правильных PDF - предупреждение
            if len(result["correct_pdfs"]) > 1:
                f.write(f"\n**️ ВНИМАНИЕ:** Найдено {len(result['correct_pdfs'])} правильных PDF! Требуется ручное разрешение конфликта.\n")
                for pdf_result in result["correct_pdfs"]:
                    f.write(f"  - {pdf_result['url']} → {Path(pdf_result.get('saved_path', '')).name}\n")
            
            f.write("\n---\n\n")
        
        f.write("## Конец отчета\n")


def main():
    """Главная функция скрипта."""
    csv_path = Path("data/deep_active_learning/references.csv")
    output_dir = Path("data/deep_active_learning/paper_pdfs")
    temp_dir = Path("data/deep_active_learning/temp_pdfs")
    report_path = Path("data/deep_active_learning/pdf_validation_report.md")
    
    # Создаем директории
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Начало валидации PDF из {csv_path}")
    logger.info(f"Правильные PDF будут сохранены в: {output_dir}")
    logger.info(f"Логи сохраняются в: {log_file}")
    
    # Читаем CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"Загружено {len(df)} статей из CSV")
    except Exception as e:
        logger.error(f"Ошибка при чтении CSV: {e}")
        return
    
    # Создаем сессию для HTTP запросов
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    # Обрабатываем каждую статью
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка статей"):
        try:
            result = process_reference(row, session, temp_dir, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке строки {idx}: {e}")
            continue
    
    # Генерируем отчет
    logger.info(f"Генерация отчета: {report_path}")
    generate_report(results, report_path)
    
    # Очищаем временную директорию
    try:
        for file in temp_dir.glob("*.pdf"):
            file.unlink()
        temp_dir.rmdir()
        logger.info("Временная директория очищена")
    except Exception as e:
        logger.warning(f"Не удалось полностью очистить временную директорию: {e}")
    
    logger.info("Валидация завершена!")
    logger.info(f"Отчет сохранен: {report_path}")
    logger.info(f"Логи: {log_file}")


if __name__ == "__main__":
    main()
