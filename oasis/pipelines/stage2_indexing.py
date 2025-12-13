"""Индексация корпуса статей в Weaviate для RAG (Шаг 2 конвейера).

Создает гибридный поисковый индекс с полной трассировкой: Paper → Chunk.
"""

import os
import warnings

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
import weaviate

# Ленивый импорт SentenceTransformer (загружается только при вызове embed_chunks)
# чтобы избежать зависания при импорте модуля

# Размерность эмбеддингов для bge-m3
BGE_M3_DIMENSION = 1024

# Подавляем предупреждения PyTorch о несовместимости CUDA capability
# Эти предупреждения появляются при проверке GPU, но мы обрабатываем их в коде
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*")
warnings.filterwarnings("ignore", message=".*NVIDIA.*with CUDA capability.*is not compatible.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")


# ============================================================================
# Pydantic модели (ADR-0003)
# ============================================================================


class Chunk(BaseModel):
    """Модель чанка текста."""

    chunk_id: str = Field(..., description="Уникальный идентификатор чанка")
    paper_id: str = Field(..., description="Идентификатор статьи")
    text: str = Field(..., description="Текст чанка")
    start_char: int = Field(..., description="Начальная позиция в тексте статьи (символы)")
    end_char: int = Field(..., description="Конечная позиция в тексте статьи (символы)")
    page_start: int | None = Field(None, description="Начальная страница чанка (опционально)")
    page_end: int | None = Field(None, description="Конечная страница чанка (опционально)")


class PaperStructure(BaseModel):
    """Модель структуры статьи."""

    paper_id: str = Field(..., description="Уникальный идентификатор статьи")
    title: str = Field(..., description="Название статьи")
    year: int | None = Field(None, description="Год публикации")
    doi: str = Field("", description="DOI статьи")
    arxiv_id: str = Field("", description="arXiv ID статьи")
    venue: str = Field("", description="Место публикации (конференция/журнал)")
    paper_type: str = Field("primary", description="Тип статьи (review/primary)")
    full_text: str = Field(..., description="Весь очищенный текст статьи")
    page_count: int = Field(..., description="Количество страниц в PDF")
    char_to_page: dict[int, int] = Field(
        default_factory=dict,
        description="Маппинг позиции символа в full_text к номеру страницы (1-based)"
    )


# ============================================================================
# Вспомогательные функции
# ============================================================================


def detect_ocr_noise(text: str, threshold: float = 0.5) -> bool:
    """Детектирует OCR-мусор по доле букв в тексте.

    Args:
        text: Текст для проверки
        threshold: Порог доли букв (по умолчанию 0.5)

    Returns:
        True если текст похож на OCR-мусор (доля букв < threshold)
    """
    if not text:
        return True

    letters = sum(c.isalpha() for c in text)
    total = len(text)
    if total == 0:
        return True

    ratio = letters / total
    return ratio < threshold


# Паттерн для шапки журнала в PDF
HEADER_RE = re.compile(r"JOURNAL OF L ATEX CLASS FILES", re.IGNORECASE)


def clean_page_text(page_text: str) -> str:
    """Очистка одной страницы PDF для RAG.

    Args:
        page_text: Текст страницы из PDF

    Returns:
        Очищенный текст страницы
    """
    if not page_text:
        return ""

    # 1. Удаляем шапки, футеры и голые номера страниц
    lines = []
    for line in page_text.splitlines():
        s = line.strip()
        if not s:
            continue

        # шапка журнала (повторяется на каждой странице)
        if HEADER_RE.search(s):
            continue

        # строка – только номер страницы
        if s.isdigit():
            continue

        lines.append(s)

    text = "\n".join(lines)

    # 2. Убираем переносы по дефису на границе строк
    # "... autoen-\ncoders ..." -> "... autoencoders ..."
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Приводим переносы строк к абзацам
    #   - двойной \n\n считаем границей абзаца
    #   - одиночные \n внутри абзаца превращаем в пробел
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "<PARA_BREAK>", text)  # временный маркер абзаца
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("<PARA_BREAK>", "\n\n")

    # 4. Нормализация некоторых символов (лигатуры и т.п.)
    text = (
        text.replace("ﬁ", "fi")
            .replace("ﬂ", "fl")
    )

    return text.strip()


# ============================================================================
# Парсинг PDF
# ============================================================================


def extract_paper_structure(
    pdf_path: str | Path,
    paper_id: str,
    references_meta: dict[str, Any],
    trace_id: str | None = None,
) -> PaperStructure:
    """Извлекает структуру статьи из PDF.

    Args:
        pdf_path: Путь к PDF файлу
        paper_id: Идентификатор статьи
        references_meta: Метаданные из references.csv
        trace_id: Идентификатор трейса для логирования

    Returns:
        PaperStructure с полным очищенным текстом статьи и маппингом позиций к страницам

    Raises:
        FileNotFoundError: Если PDF файл не найден
        IOError: Если не удалось прочитать PDF
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

    logger.info(f"{trace_prefix}Извлечение структуры из PDF: {path.name}")

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        
        # Очищаем каждую страницу и создаем границы страниц для маппинга
        cleaned_pages = []
        page_boundaries = []  # Границы страниц: (page_num, start_pos, end_pos)
        current_pos = 0
        
        for page_num, page in enumerate(reader.pages, 1):  # page_num начинается с 1
            raw = page.extract_text() or ""
            cleaned = clean_page_text(raw)
            if cleaned:  # Пропускаем пустые страницы
                cleaned_pages.append(cleaned)
                page_start = current_pos
                page_end = current_pos + len(cleaned)
                page_boundaries.append((page_num, page_start, page_end))
                current_pos = page_end
                # Добавляем разделитель между страницами (\n\n = 2 символа)
                if page_num < len(reader.pages):  # Не добавляем после последней страницы
                    current_pos += 2
        
        # Объединяем все страницы с разделителем \n\n между страницами
        full_text = "\n\n".join(cleaned_pages)
        
        # Освобождаем память
        del cleaned_pages
        import gc
        gc.collect()
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка чтения PDF {path.name}: {e}")
        raise IOError(f"Не удалось прочитать PDF: {e}") from e

    # Детекция OCR-мусора
    if detect_ocr_noise(full_text):
        logger.warning(f"{trace_prefix}PDF {path.name} похож на OCR-мусор, пропускаем")
        raise ValueError(f"OCR-мусор детектирован в {path.name}")

    # Опционально: удаление раздела References/Bibliography
    references_pattern = re.compile(
        r"(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)", re.IGNORECASE | re.MULTILINE
    )
    references_match = references_pattern.search(full_text)
    if references_match:
        references_start_pos = references_match.start()
        full_text = full_text[:references_start_pos]
        logger.debug(f"{trace_prefix}Найден раздел References, обрезаем текст")

    # Опционально: удаление раздела Acknowledgments
    acknowledgments_pattern = re.compile(
        r"(?:^|\n)\s*(?:acknowledgments?|acknowledgements?)\s*(?:\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    acknowledgments_match = acknowledgments_pattern.search(full_text)
    if acknowledgments_match:
        acknowledgments_start_pos = acknowledgments_match.start()
        full_text = full_text[:acknowledgments_start_pos]
        logger.debug(f"{trace_prefix}Найден раздел Acknowledgments, обрезаем текст")

    logger.info(f"{trace_prefix}Извлечен текст из {page_count} страниц PDF {path.name}, длина: {len(full_text)} символов")

    # Создаем маппинг позиций символов к номерам страниц
    # Используем границы страниц, учитывая обрезку текста
    char_to_page = {}
    for page_num, page_start, page_end in page_boundaries:
        # Учитываем обрезку текста
        actual_end = min(page_end, len(full_text))
        # Заполняем маппинг для всех символов страницы
        for pos in range(page_start, actual_end):
            char_to_page[pos] = page_num
        # Разделитель между страницами (\n\n) также относится к текущей странице
        if page_end < len(full_text):
            char_to_page[page_end] = page_num  # Первый \n разделителя
        if page_end + 1 < len(full_text):
            char_to_page[page_end + 1] = page_num  # Второй \n разделителя

    return PaperStructure(
        paper_id=paper_id,
        title=references_meta.get("title", ""),
        year=references_meta.get("year"),
        doi=references_meta.get("doi", ""),
        arxiv_id=references_meta.get("arxiv_id", ""),
        venue=references_meta.get("venue", ""),
        paper_type=references_meta.get("paper_type", "primary"),
        full_text=full_text,
        page_count=page_count,
        char_to_page=char_to_page,
    )


# ============================================================================
# Чанкирование
# ============================================================================


def chunk_paper(
    paper_text: str,
    paper_id: str,
    chunk_size: int = 1000,
    overlap: int = 300,
    char_to_page: dict[int, int] | None = None,
    trace_id: str | None = None,
) -> list[Chunk]:
    """Разбивает текст статьи на чанки с перекрытием.

    Args:
        paper_text: Текст статьи для чанкирования
        paper_id: Идентификатор статьи
        chunk_size: Размер чанка в символах
        overlap: Размер перекрытия в символах
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список Chunk объектов
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    chunks = []

    if not paper_text:
        return chunks

    start = 0
    chunk_num = 0
    prev_start = -1  # Для отслеживания дубликатов

    # Множество для отслеживания уникальности позиций (предотвращение дубликатов)
    seen_positions = set()
    
    while start < len(paper_text):
        end = min(start + chunk_size, len(paper_text))
        chunk_text = paper_text[start:end].strip()

        # Пропускаем слишком короткие чанки (минимальный размер 200 символов)
        if len(chunk_text) < 200:
            break

        # Проверка на дубликаты: если start не изменился или откатился назад, прекращаем
        if start <= prev_start:
            logger.warning(
                f"{trace_prefix}Обнаружен дубликат чанка (start={start}, prev_start={prev_start}), "
                f"прекращаем чанкирование для {paper_id}"
            )
            break

        # Проверка на дубликаты по позициям (защита от полных дубликатов)
        position_key = (start, end)
        if position_key in seen_positions:
            logger.warning(
                f"{trace_prefix}Обнаружен дубликат позиции (start={start}, end={end}), "
                f"пропускаем чанк для {paper_id}"
            )
            # Пропускаем этот чанк и переходим к следующему
            start = end - overlap
            continue
        
        seen_positions.add(position_key)

        chunk_id = f"{paper_id}::chunk_{chunk_num:03d}"

        # Вычисляем номера страниц на основе маппинга
        page_start = None
        page_end = None
        if char_to_page:
            # Берем страницу для начальной позиции
            page_start = char_to_page.get(start)
            # Берем страницу для конечной позиции (или последней позиции чанка)
            page_end = char_to_page.get(end - 1) if end > 0 else char_to_page.get(start)
            # Если не нашли точную страницу, ищем ближайшую
            if page_start is None:
                # Ищем ближайшую позицию с известной страницей
                for offset in range(min(100, len(paper_text) - start)):
                    if start + offset in char_to_page:
                        page_start = char_to_page[start + offset]
                        break
            if page_end is None:
                # Ищем ближайшую позицию с известной страницей
                for offset in range(min(100, end)):
                    if end - 1 - offset in char_to_page:
                        page_end = char_to_page[end - 1 - offset]
                        break
            # Если все еще не нашли, используем page_start для обоих
            if page_end is None:
                page_end = page_start

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                page_start=page_start,
                page_end=page_end,
            )
        )

        chunk_num += 1
        prev_start = start  # Сохраняем текущий start перед обновлением
        start = end - overlap  # Перекрытие
        
        # Дополнительная проверка: если достигли конца текста и следующий чанк будет дубликатом
        if end >= len(paper_text):
            break

    logger.debug(f"{trace_prefix}Создано {len(chunks)} чанков из статьи {paper_id}")

    return chunks


# ============================================================================
# Векторизация
# ============================================================================


# Глобальная переменная для переиспользования модели
# Эмбеддинги хранятся в Weaviate, кэш не нужен
_embedding_model: Any = None


def _check_cuda_compatibility(trace_id: str | None = None) -> tuple[bool, str]:
    """Проверяет совместимость CUDA с текущим PyTorch и GPU.
    
    Проверяет capability GPU и совместимость с PyTorch ДО попытки выполнения операций,
    чтобы избежать ошибок и предупреждений. Подавляет предупреждения PyTorch о несовместимости.
    
    Args:
        trace_id: Идентификатор трейса для логирования
        
    Returns:
        Кортеж (is_compatible, device_name), где:
        - is_compatible: True если CUDA совместима и работает
        - device_name: Название устройства ("cuda" или "cpu")
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    try:
        import torch
        import warnings
        
        # Подавляем предупреждения PyTorch о несовместимости capability
        # Устанавливаем фильтр до первого обращения к torch.cuda
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            if not torch.cuda.is_available():
                logger.debug(f"{trace_prefix}CUDA недоступна (torch.cuda.is_available() = False)")
                return False, "cpu"
        
            # Получаем информацию о GPU
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.debug(f"{trace_prefix}CUDA доступна, но нет GPU устройств")
                return False, "cpu"
            
            # Получаем информацию о GPU с подавлением предупреждений PyTorch
            try:
                device_name = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                capability_str = f"{capability[0]}.{capability[1]}"
            
                logger.info(
                    f"{trace_prefix}Обнаружен GPU: {device_name}, "
                    f"capability={capability_str}, devices={device_count}"
                )
                
                # Проверяем минимальные требования PyTorch к capability
                # PyTorch 2.2.0+ с CUDA 11.8 поддерживает capability >= 3.5
                # Но для надежности проверяем через тестовую операцию
                # Не блокируем использование GPU на основе capability, если PyTorch поддерживает его
                
                # Если capability подходит, проверяем работоспособность через безопасный тест
                # Подавляем все предупреждения при тестировании
                try:
                    # Пытаемся создать небольшой тензор на GPU
                    test_tensor = torch.zeros(1, device="cuda")
                    # Пытаемся выполнить простую операцию
                    test_result = test_tensor + 1
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    
                    logger.info(f"{trace_prefix}CUDA совместима и работает, используем GPU")
                    return True, "cuda"
                except RuntimeError as runtime_e:
                    error_msg = str(runtime_e).lower()
                    if "kernel" in error_msg or "capability" in error_msg or "no kernel image" in error_msg:
                        logger.info(
                            f"{trace_prefix}GPU {device_name} несовместим с текущей версией PyTorch "
                            f"(capability {capability_str}). Используем CPU."
                        )
                        return False, "cpu"
                    else:
                        # Другая ошибка - пробрасываем
                        raise
                except Exception as test_e:
                    # Любая другая ошибка при тестировании
                    logger.warning(
                        f"{trace_prefix}Ошибка при тестировании CUDA, используем CPU: {test_e}"
                    )
                    return False, "cpu"
                    
            except Exception as e:
                # Ошибка при получении информации о GPU
                logger.warning(
                    f"{trace_prefix}Не удалось получить информацию о GPU, используем CPU: {e}"
                )
                return False, "cpu"
            
    except ImportError:
        logger.debug(f"{trace_prefix}PyTorch не установлен, используем CPU")
        return False, "cpu"
    except Exception as e:
        logger.warning(f"{trace_prefix}Ошибка при проверке CUDA, используем CPU: {e}")
        return False, "cpu"


def embed_chunks(
    chunks: list[Chunk],
    model_path: str | Path = "models/bge-m3",
    batch_size: int = 8,
    trace_id: str | None = None,
) -> list[np.ndarray]:
    """Векторизует чанки с помощью локальной модели bge-m3.

    Args:
        chunks: Список чанков для векторизации
        model_path: Путь к локальной модели или название модели
        batch_size: Размер батча для обработки
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список numpy массивов с эмбеддингами
    """
    global _embedding_model
    
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    if not chunks:
        return []

    # Проверяем существование локальной модели (ADR-0007)
    # Преобразуем путь к модели в абсолютный, если это локальный путь
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        model_name = str(model_path_obj.absolute())
    else:
        # Если относительный путь не найден, пробуем относительно корня проекта
        project_root = Path(__file__).parent.parent.parent
        project_model_path = project_root / model_path
        if project_model_path.exists():
            model_name = str(project_model_path.absolute())
        else:
            # Если и локальный путь не найден, используем как есть (может быть HuggingFace ID)
            model_name = str(model_path)

    # Загружаем модель один раз и переиспользуем
    if _embedding_model is None:
        if model_path_obj.exists():
            logger.info(f"{trace_prefix}Загрузка локальной модели: {model_name}")
        else:
            logger.info(f"{trace_prefix}Загрузка модели из HuggingFace: {model_name}")
        
        from sentence_transformers import SentenceTransformer
        
        # Проверяем совместимость CUDA перед загрузкой модели
        cuda_compatible, device = _check_cuda_compatibility(trace_id)
        
        logger.info(f"{trace_prefix}Инициализация SentenceTransformer (device={device})...")
        
        try:
            _embedding_model = SentenceTransformer(model_name, device=device)
            # Проверяем, что модель действительно работает на выбранном устройстве
            try:
                _embedding_model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
                logger.info(f"{trace_prefix}Модель загружена успешно на {device}")
            except Exception as test_e:
                # Если тестовый encode не работает на CUDA, переключаемся на CPU
                if device == "cuda" and ("cuda" in str(test_e).lower() or "kernel" in str(test_e).lower()):
                    logger.warning(
                        f"{trace_prefix}CUDA ошибка при тестировании модели, переключаемся на CPU: {test_e}"
                    )
                    _embedding_model = SentenceTransformer(model_name, device="cpu")
                    logger.info(f"{trace_prefix}Модель перезагружена на CPU")
                else:
                    raise
        except Exception as e:
            if device == "cuda":
                logger.warning(f"{trace_prefix}Ошибка при загрузке на CUDA, пробуем CPU: {e}")
                _embedding_model = SentenceTransformer(model_name, device="cpu")
                logger.info(f"{trace_prefix}Модель загружена на CPU, переносим на GPU...")
                # Переносим модель на GPU после загрузки на CPU
                try:
                    _embedding_model = _embedding_model.to("cuda")
                    logger.info(f"{trace_prefix}Модель перенесена на GPU")
                except Exception as gpu_e:
                    logger.warning(f"{trace_prefix}Не удалось перенести модель на GPU, используем CPU: {gpu_e}")
            else:
                raise
    
    model = _embedding_model

    # Извлекаем тексты чанков (кэш отключен)
    texts = [chunk.text for chunk in chunks]

    # Векторизуем все тексты
    new_embeddings = []
    if texts:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(
            f"{trace_prefix}Векторизация {len(texts)} текстов "
            f"(батч={batch_size}, всего батчей: {total_batches})"
        )
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Векторизация чанков",
            unit="батч",
        ):
            batch_texts = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.debug(f"{trace_prefix}Обработка батча {batch_num}/{total_batches} ({len(batch_texts)} текстов)")
            
            try:
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                new_embeddings.extend(batch_embeddings)
                logger.debug(f"{trace_prefix}Батч {batch_num}/{total_batches} обработан")
            except Exception as encode_error:
                # Если возникла CUDA ошибка при encode, перезагружаем модель на CPU
                if "cuda" in str(encode_error).lower() or "kernel" in str(encode_error).lower():
                    logger.warning(f"{trace_prefix}CUDA ошибка при векторизации батча {batch_num}, переключаемся на CPU: {encode_error}")
                    _embedding_model = SentenceTransformer(model_name, device="cpu")
                    model = _embedding_model
                    logger.info(f"{trace_prefix}Модель перезагружена на CPU, повторяем батч {batch_num}")
                    # Повторяем батч на CPU
                    batch_embeddings = model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    new_embeddings.extend(batch_embeddings)
                    logger.debug(f"{trace_prefix}Батч {batch_num}/{total_batches} обработан на CPU")
                else:
                    # Другая ошибка - пробрасываем дальше
                    raise

    logger.info(
        f"{trace_prefix}Вычислено {len(new_embeddings)} эмбеддингов "
        f"размерности {new_embeddings[0].shape[0] if new_embeddings else 0}"
    )
    
    # Освобождаем память от промежуточных списков
    del texts

    return new_embeddings




# ============================================================================
# Weaviate схема
# ============================================================================


def create_weaviate_schema(
    client: weaviate.Client,
    delete_existing: bool = True,
    trace_id: str | None = None,
) -> None:
    """Создает схему Weaviate с классами Paper и Chunk (parent-child).

    Args:
        client: Weaviate клиент
        delete_existing: Удалить существующие классы если True
        trace_id: Идентификатор трейса для логирования
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    # Удаление существующих классов (включая устаревший Section)
    # ВАЖНО: При удалении класса все объекты этого класса автоматически удаляются
    if delete_existing:
        # Удаляем в правильном порядке: сначала Chunk (зависит от Paper), затем Paper, затем Section
        for class_name in ["Chunk", "Paper", "Section"]:
            try:
                # Проверяем, существует ли класс перед удалением
                schema = client.schema.get()
                existing_classes = [c.get("class") for c in schema.get("classes", [])]
                
                if class_name in existing_classes:
                    client.schema.delete_class(class_name)
                    logger.info(f"{trace_prefix}Удален класс {class_name} (все объекты также удалены)")
                else:
                    logger.debug(f"{trace_prefix}Класс {class_name} не существует, пропускаем")
            except Exception as e:
                logger.warning(f"{trace_prefix}Ошибка при удалении класса {class_name}: {e}")
                # Продолжаем работу, даже если удаление не удалось

    # Создание класса Paper
    paper_class = {
        "class": "Paper",
        "description": "Научная статья из корпуса",
        "vectorizer": "none",
        "properties": [
            {"name": "paper_id", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "year", "dataType": ["int"]},
            {"name": "doi", "dataType": ["text"]},
            {"name": "arxiv_id", "dataType": ["text"]},
            {"name": "venue", "dataType": ["text"]},
            {"name": "paper_type", "dataType": ["text"]},
        ],
    }

    try:
        client.schema.create_class(paper_class)
        logger.info(f"{trace_prefix}Создан класс Paper")
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка создания класса Paper: {e}")
        raise

    # Создание класса Chunk с ссылкой на Paper
    chunk_class = {
        "class": "Chunk",
        "description": "Текстовый чанк из статьи",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine",
        },
        "properties": [
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "paper_id", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"], "indexInverted": True},  # BM25
            {"name": "start_char", "dataType": ["int"]},
            {"name": "end_char", "dataType": ["int"]},
            {"name": "page_start", "dataType": ["int"]},
            {"name": "page_end", "dataType": ["int"]},
        ],
    }

    try:
        client.schema.create_class(chunk_class)
        logger.info(f"{trace_prefix}Создан класс Chunk")

        # Добавляем ссылку на Paper после создания класса
        parent_property = {
            "dataType": ["Paper"],
            "description": "Ссылка на родительскую статью",
            "name": "parent",
        }
        client.schema.property.create("Chunk", parent_property)
        logger.info(f"{trace_prefix}Добавлена ссылка parent в Chunk")
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка создания класса Chunk: {e}")
        raise

    logger.success(f"{trace_prefix}Схема Weaviate создана успешно")


# ============================================================================
# Загрузка данных в Weaviate
# ============================================================================


def index_corpus(
    pdf_dir: str | Path,
    references_csv: str | Path,
    weaviate_url: str,
    trace_id: str | None = None,
        batch_size: int = 100,
        embedding_batch_size: int = 8,
    chunk_size: int = 1000,
    chunk_overlap: int = 300,
    model_path: str | Path = "models/bge-m3",
    delete_existing: bool = True,
) -> dict[str, Any]:
    """Индексирует весь корпус PDF в Weaviate.

    Args:
        pdf_dir: Директория с PDF файлами
        references_csv: Путь к CSV с метаданными
        weaviate_url: URL Weaviate сервера
        trace_id: Идентификатор трейса
        batch_size: Размер батча для загрузки в Weaviate (влияет на память при загрузке)
        embedding_batch_size: Размер батча для генерации эмбеддингов (влияет на память GPU/CPU)
        chunk_size: Размер чанка в символах (влияет на количество чанков и память)
        chunk_overlap: Размер перекрытия в символах (влияет на количество чанков)
        model_path: Путь к модели эмбеддингов
        delete_existing: Удалить существующие классы и данные перед индексацией (по умолчанию True)

    Returns:
        Словарь со статистикой: papers_processed, chunks_processed,
        papers_failed, failed_files
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    logger.info(f"{trace_prefix}Начало индексации корпуса")

    # Подключение к Weaviate
    try:
        client = weaviate.Client(weaviate_url)
        logger.info(f"{trace_prefix}Подключено к Weaviate: {weaviate_url}")
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка подключения к Weaviate: {e}")
        raise

    # Создание схемы
    create_weaviate_schema(client, delete_existing=delete_existing, trace_id=trace_id)

    # Загрузка метаданных
    pdf_dir_path = Path(pdf_dir)
    references_path = Path(references_csv)

    if not references_path.exists():
        raise FileNotFoundError(f"Файл метаданных не найден: {references_csv}")

    df = pd.read_csv(references_path)
    logger.info(f"{trace_prefix}Загружено {len(df)} записей из {references_csv}")

    # Создаем словарь метаданных по ref_number
    metadata_dict = {}
    for _, row in df.iterrows():
        ref_number = str(row.get("ref_number", ""))
        if ref_number:
            metadata_dict[ref_number] = {
                "title": str(row.get("title", "")),
                "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
                "doi": str(row.get("doi", "")),
                "arxiv_id": str(row.get("arxiv_id", "")),
                "venue": str(row.get("venue", "")),
                "paper_type": "primary",  # По умолчанию, можно определить по названию
            }

    # Собираем все PDF файлы и сортируем по ref_number
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    
    # Сортируем по числовому значению ref_number в начале имени файла
    # Формат: {ref_number}-{arxiv_id}.pdf
    def get_ref_number(pdf_path: Path) -> int:
        """Извлекает ref_number из имени файла для сортировки."""
        try:
            parts = pdf_path.stem.split("-", 1)
            if len(parts) >= 1:
                return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 999999  # Файлы без числового префикса в конец
    
    pdf_files.sort(key=get_ref_number)
    logger.info(f"{trace_prefix}Найдено {len(pdf_files)} PDF файлов (отсортировано по ref_number)")

    # Статистика
    papers_processed = 0
    chunks_processed = 0
    papers_failed = 0
    failed_files = []

    # Словарь для хранения UUID объектов Paper (для ссылок parent-child)
    paper_uuids: dict[str, str] = {}  # paper_id -> uuid

    # Все объекты загружаются сразу после создания, не накапливаются в памяти
    # Это критично для экономии памяти при больших PDF

    # Обработка каждого PDF
    for pdf_path in tqdm(pdf_files, desc="Обработка PDF", unit="файл"):
        try:
            # Извлекаем paper_id из имени файла: {ref_number}-{arxiv_id}.pdf
            pdf_name = pdf_path.stem
            parts = pdf_name.split("-", 1)
            if len(parts) < 2:
                logger.warning(f"{trace_prefix}Неверный формат имени файла: {pdf_path.name}")
                papers_failed += 1
                failed_files.append(pdf_path.name)
                continue

            ref_number = parts[0]
            paper_id = pdf_name

            # Получаем метаданные
            references_meta = metadata_dict.get(ref_number, {})
            if not references_meta:
                logger.warning(f"{trace_prefix}Метаданные не найдены для ref_number={ref_number}")
                references_meta = {
                    "title": "",
                    "year": None,
                    "doi": "",
                    "arxiv_id": "",
                    "venue": "",
                    "paper_type": "primary",
                }

            # Парсинг структуры
            logger.info(f"{trace_prefix}Парсинг структуры PDF: {pdf_path.name}")
            paper_structure = extract_paper_structure(
                pdf_path=pdf_path,
                paper_id=paper_id,
                references_meta=references_meta,
                trace_id=trace_id,
            )
            logger.info(
                f"{trace_prefix}Извлечен текст из {paper_structure.page_count} страниц, "
                f"длина: {len(paper_structure.full_text)} символов"
            )

            # Создаем объект Paper и сразу загружаем в Weaviate (нужен для ссылок в Chunk)
            paper_obj = {
                "paper_id": paper_structure.paper_id,
                "title": paper_structure.title,
                "year": paper_structure.year,
                "doi": paper_structure.doi,
                "arxiv_id": paper_structure.arxiv_id,
                "venue": paper_structure.venue,
                "paper_type": paper_structure.paper_type,
            }
            
            # Загружаем Paper сразу (нужен для ссылок в Chunk)
            logger.debug(f"{trace_prefix}Загрузка Paper '{paper_structure.paper_id}' в Weaviate")
            with client.batch as batch:
                batch.batch_size = batch_size
                paper_uuid = batch.add_data_object(paper_obj, "Paper")
                if paper_uuid:
                    paper_uuids[paper_structure.paper_id] = paper_uuid
                # Явно отправляем batch, чтобы Paper был доступен для ссылок
                batch.flush()

            # Чанкирование всего текста статьи
            logger.debug(f"{trace_prefix}Чанкирование текста статьи '{paper_structure.paper_id}'")
            chunks = chunk_paper(
                paper_text=paper_structure.full_text,
                paper_id=paper_structure.paper_id,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                char_to_page=paper_structure.char_to_page,
                trace_id=trace_id,
            )
            logger.info(f"{trace_prefix}Создано {len(chunks)} чанков из статьи {paper_structure.paper_id}")

            # Освобождаем память от full_text сразу после чанкирования
            del paper_structure.full_text
            import gc
            gc.collect()

            # Векторизация всех чанков
            logger.debug(f"{trace_prefix}Векторизация {len(chunks)} чанков")
            embeddings = embed_chunks(
                chunks=chunks,
                model_path=model_path,
                batch_size=embedding_batch_size,
                trace_id=trace_id,
            )
            logger.info(f"{trace_prefix}Вычислено {len(embeddings)} эмбеддингов")

            # Загрузка всех чанков в Weaviate
            logger.debug(f"{trace_prefix}Загрузка {len(chunks)} чанков в Weaviate")
            chunk_uuids_for_refs = []  # Список (chunk_uuid, paper_id) для добавления ссылок
            
            # Проверка на дубликаты chunk_id перед загрузкой
            chunk_ids_seen = set()
            duplicate_chunk_ids = []
            for chunk in chunks:
                if chunk.chunk_id in chunk_ids_seen:
                    duplicate_chunk_ids.append(chunk.chunk_id)
                else:
                    chunk_ids_seen.add(chunk.chunk_id)
            
            if duplicate_chunk_ids:
                logger.error(
                    f"{trace_prefix}Обнаружены дубликаты chunk_id в статье {paper_structure.paper_id}: "
                    f"{len(duplicate_chunk_ids)} дубликатов. Это критическая ошибка!"
                )
                for dup_id in duplicate_chunk_ids[:5]:
                    logger.error(f"{trace_prefix}  - Дубликат: {dup_id}")
                raise ValueError(f"Дубликаты chunk_id обнаружены для статьи {paper_structure.paper_id}")
            
            with client.batch as batch_chunks:
                batch_chunks.batch_size = batch_size
                
                for chunk, embedding in zip(chunks, embeddings):
                    # Проверка формата chunk_id (не должно быть старых форматов с sec_)
                    if "::sec_" in chunk.chunk_id:
                        logger.error(
                            f"{trace_prefix}Обнаружен старый формат chunk_id: {chunk.chunk_id}. "
                            f"Это должно быть невозможно в текущей реализации!"
                        )
                        continue
                    
                    chunk_obj = {
                        "chunk_id": chunk.chunk_id,
                        "paper_id": chunk.paper_id,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    }
                    
                    try:
                        # Преобразуем вектор в список
                        vector_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        
                        if not isinstance(vector_list, list) or len(vector_list) == 0:
                            logger.error(f"{trace_prefix}Невалидный вектор для чанка {chunk.chunk_id}")
                            continue
                        
                        uuid = batch_chunks.add_data_object(chunk_obj, "Chunk", vector=vector_list)
                        
                        if uuid:
                            chunk_uuids_for_refs.append((uuid, chunk.paper_id))
                            logger.debug(f"{trace_prefix}Чанк {chunk.chunk_id} добавлен в batch, UUID: {uuid[:8]}...")
                        else:
                            logger.error(f"{trace_prefix}UUID не получен для чанка {chunk.chunk_id}")
                    except Exception as e:
                        logger.error(f"{trace_prefix}Ошибка добавления чанка {chunk.chunk_id} в batch: {e}")
                
                # Отправляем batch с чанками
                if len(chunks) > 0:
                    logger.info(f"{trace_prefix}Отправка batch с {len(chunks)} чанками в Weaviate")
                    try:
                        batch_chunks.flush()
                        logger.info(f"{trace_prefix}Batch отправлен успешно, загружено {len(chunks)} чанков")
                    except Exception as e:
                        logger.error(f"{trace_prefix}Ошибка при вызове flush(): {e}")
                        raise
                    
                    # Проверяем ошибки batch
                    if hasattr(batch_chunks, 'errors') and batch_chunks.errors:
                        logger.error(f"{trace_prefix}Ошибки при загрузке batch: {len(batch_chunks.errors)} ошибок")
                        for error in batch_chunks.errors[:5]:
                            logger.error(f"{trace_prefix}  - {error}")

            # Сохраняем количество чанков перед освобождением памяти
            num_chunks_this_paper = len(chunks)
            
            # Добавляем ссылки Chunk -> Paper (опционально, не критично для RAG)
            if chunk_uuids_for_refs and paper_structure.paper_id in paper_uuids:
                paper_uuid = paper_uuids[paper_structure.paper_id]
                if paper_uuid and isinstance(paper_uuid, str) and len(paper_uuid) >= 32:
                    logger.debug(f"{trace_prefix}Добавление ссылок Chunk -> Paper для {len(chunk_uuids_for_refs)} чанков")
                    # Ссылки можно добавить позже, они не обязательны для гибридного поиска

            # Освобождаем память
            del chunks, embeddings, chunk_uuids_for_refs
            gc.collect()

            chunks_processed += num_chunks_this_paper
            papers_processed += 1
            
            logger.info(
                f"{trace_prefix}PDF {pdf_path.name} обработан: "
                f"{num_chunks_this_paper} чанков (всего: {chunks_processed} чанков)"
            )
            
            # Освобождаем память от paper_structure
            del paper_structure

            # Старый код обработки секций удален - теперь обрабатываем весь текст статьи целиком

        except ValueError as e:
            # OCR-мусор или другие ошибки парсинга
            logger.warning(f"{trace_prefix}Пропущен PDF {pdf_path.name}: {e}")
            papers_failed += 1
            failed_files.append(pdf_path.name)
        except Exception as e:
            logger.error(f"{trace_prefix}Ошибка обработки PDF {pdf_path.name}: {e}")
            papers_failed += 1
            failed_files.append(pdf_path.name)
        
        # Явное освобождение памяти после обработки каждого PDF
        import gc
        gc.collect()

    stats = {
        "papers_processed": papers_processed,
        "chunks_processed": chunks_processed,
        "papers_failed": papers_failed,
        "failed_files": failed_files,
    }

    logger.success(f"{trace_prefix}Индексация завершена: {stats}")

    return stats


# ============================================================================
# Гибридный поиск
# ============================================================================


def hybrid_search(
    query: str,
    weaviate_client: weaviate.Client,
    top_k: int = 10,
    alpha: float = 0.5,
    filters: dict[str, Any] | None = None,
    trace_id: str | None = None,
    model_path: str | Path = "models/bge-m3",
) -> list[dict[str, Any]]:
    """Выполняет гибридный поиск в Weaviate (BM25 + векторный).

    Args:
        query: Текстовый запрос
        weaviate_client: Weaviate клиент
        top_k: Количество результатов
        alpha: Вес векторного поиска (0.0 = только BM25, 1.0 = только векторный)
        filters: Фильтры по метаданным (paper_type, year)
        trace_id: Идентификатор трейса
        model_path: Путь к модели эмбеддингов

    Returns:
        Список словарей с результатами: chunk_id, text, paper_id, start_char, end_char,
        page_start, page_end, score
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    # Ленивый импорт SentenceTransformer (избегаем зависания при импорте модуля)
    from sentence_transformers import SentenceTransformer
    
    # Векторизация запроса
    # Преобразуем путь к модели в абсолютный, если это локальный путь
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        model_name = str(model_path_obj.absolute())
    else:
        # Если относительный путь не найден, пробуем относительно корня проекта
        project_root = Path(__file__).parent.parent.parent
        project_model_path = project_root / model_path
        if project_model_path.exists():
            model_name = str(project_model_path.absolute())
        else:
            # Если и локальный путь не найден, используем как есть (может быть HuggingFace ID)
            model_name = str(model_path)

    # Используем глобальную модель для переиспользования (избегаем перезагрузки и CUDA OOM)
    global _embedding_model
    
    # Загружаем модель один раз и переиспользуем
    if _embedding_model is None:
        # Проверяем совместимость CUDA перед загрузкой модели
        cuda_compatible, device = _check_cuda_compatibility(trace_id)
        logger.info(f"{trace_prefix}Загрузка модели для векторизации запроса (device={device})...")
        
        try:
            _embedding_model = SentenceTransformer(model_name, device=device)
            # Проверяем работоспособность на выбранном устройстве
            try:
                _embedding_model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
            except Exception as test_e:
                if device == "cuda" and ("cuda" in str(test_e).lower() or "kernel" in str(test_e).lower()):
                    logger.warning(f"{trace_prefix}CUDA ошибка при тестировании, используем CPU: {test_e}")
                    _embedding_model = SentenceTransformer(model_name, device="cpu")
                else:
                    raise
        except Exception as e:
            if device == "cuda":
                logger.warning(f"{trace_prefix}Ошибка при загрузке на CUDA, используем CPU: {e}")
                _embedding_model = SentenceTransformer(model_name, device="cpu")
            else:
                raise
    else:
        pass
        # logger.debug(f"{trace_prefix}Используем уже загруженную модель для векторизации запроса")
    
    model = _embedding_model
    
    # Векторизуем запрос с обработкой ошибок
    try:
        query_embedding = model.encode(query, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    except Exception as encode_error:
        if "cuda" in str(encode_error).lower() or "out of memory" in str(encode_error).lower():
            logger.warning(f"{trace_prefix}CUDA ошибка при векторизации запроса, перезагружаем модель на CPU: {encode_error}")
            # Перезагружаем модель на CPU при ошибке памяти
            _embedding_model = SentenceTransformer(model_name, device="cpu")
            model = _embedding_model
            query_embedding = model.encode(query, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
        else:
            raise
    query_vector = query_embedding[0].tolist() if len(query_embedding.shape) > 1 else query_embedding.tolist()

    # Построение запроса (старый API Weaviate)
    query_builder = (
        weaviate_client.query.get(
            "Chunk",
            ["chunk_id", "text", "paper_id", "start_char", "end_char", "page_start", "page_end"]
        )
        .with_additional(["score"])
    )

    # Применение фильтров (paper_type, year)
    if filters:
        where_clauses = []
        if "paper_type" in filters:
            where_clauses.append({
                "path": ["paper_type"],
                "operator": "Equal",
                "valueText": filters["paper_type"],
            })
        if "year" in filters:
            where_clauses.append({
                "path": ["year"],
                "operator": "Equal",
                "valueInt": filters["year"],
            })

        if where_clauses:
            if len(where_clauses) == 1:
                query_builder = query_builder.with_where(where_clauses[0])
            else:
                # Объединяем фильтры через AND
                where_filter = {"operator": "And", "operands": where_clauses}
                query_builder = query_builder.with_where(where_filter)

    # Гибридный поиск
    result = (
        query_builder.with_hybrid(query=query, vector=query_vector, alpha=alpha)
        .with_limit(top_k)
        .do()
    )

    # Обработка результатов
    chunks = []
    if "data" in result and "Get" in result["data"] and "Chunk" in result["data"]["Get"]:
        for item in result["data"]["Get"]["Chunk"]:
            additional = item.get("_additional", {})
            chunks.append({
                "chunk_id": item.get("chunk_id", ""),
                "text": item.get("text", ""),
                "paper_id": item.get("paper_id", ""),
                "start_char": item.get("start_char", 0),
                "end_char": item.get("end_char", 0),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "score": additional.get("score", 0.0),
            })

    # logger.info(f"{trace_prefix}Найдено {len(chunks)} чанков по запросу")

    return chunks

