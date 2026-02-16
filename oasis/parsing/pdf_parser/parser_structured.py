"""
Структурированный парсер PDF с использованием get_text("dict").
Использует координаты, размеры шрифтов и типы блоков для лучшего определения структуры.
"""

from pathlib import Path
import re
from typing import List, Optional, Set, Tuple

import fitz

from oasis.parsing.pdf_parser.config import ParsePDFConfig
from oasis.parsing.pdf_parser.structured import (
    TextBlock,
    extract_structured_blocks,
    merge_nearby_blocks,
    extract_table_blocks_structured,
)
from oasis.parsing.pdf_parser.figures import RE_FIG_CAPTION, caption_block_end
from oasis.parsing.pdf_parser.headers import collect_running_headers_footers
from oasis.parsing.pdf_parser.postprocess import postprocess_text
from oasis.parsing.pdf_parser.references import RE_REFERENCES, cut_references_section
from oasis.parsing.pdf_parser.tables import (
    RE_TABLE_CAPTION,
    format_table_rag,
    parse_table_from_lines,
)


def filter_blocks_by_coordinates(
    blocks: List[TextBlock],
    remove_keys: Set[str],
    top_margin_pt: float = 72.0,
    bottom_margin_pt: float = 72.0,
    page_height: float = 792.0,
) -> List[TextBlock]:
    """
    Удаляет блоки, которые находятся в области headers/footers.
    
    Args:
        blocks: Список блоков
        remove_keys: Множество нормализованных ключей для удаления
        top_margin_pt: Верхняя зона в пунктах
        bottom_margin_pt: Нижняя зона в пунктах
        page_height: Высота страницы
    
    Returns:
        Отфильтрованный список блоков
    """
    from oasis.parsing.pdf_parser.headers import _norm_hf_line
    
    filtered = []
    for block in blocks:
        if block.block_type != 0:  # Пропускаем изображения
            continue
        
        # Проверяем позицию блока
        is_header = block.y0 <= top_margin_pt
        is_footer = block.y1 >= (page_height - bottom_margin_pt)
        
        if is_header or is_footer:
            # Проверяем, нужно ли удалить этот блок
            should_remove = False
            for line in block.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                key = _norm_hf_line(line)
                if key in remove_keys:
                    should_remove = True
                    break
            
            if should_remove:
                continue
        
        filtered.append(block)
    
    return filtered


def is_formula_line(s: str, cfg: ParsePDFConfig) -> bool:
    """Определяет, похожа ли строка на математическую формулу."""
    from oasis.parsing.pdf_parser.figures import looks_like_paragraph

    s = (s or "").strip()
    if not s or len(s) < 8:
        return False
    if RE_TABLE_CAPTION.match(s) or RE_FIG_CAPTION.match(s):
        return False
    if looks_like_paragraph(s):
        return False

    latex_markers = [
        "\\frac", "\\sum", "\\int", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta",
        "\\lambda", "\\mu", "\\sigma", "\\phi", "\\psi", "\\omega", "\\mathbb",
        "\\mathbf", "\\mathrm", "\\mathcal", "\\cdot", "\\times", "\\approx",
        "\\leq", "\\geq", "\\neq",
    ]
    if any(m in s for m in latex_markers):
        return True

    symbols = set("=+-*/^<>_~|()[]{}")
    non_space_len = sum(not c.isspace() for c in s)
    if non_space_len == 0:
        return False

    symbol_count = sum(1 for c in s if c in symbols)
    digit_count = sum(c.isdigit() for c in s)
    symbol_ratio = symbol_count / non_space_len
    digit_ratio = digit_count / non_space_len

    tokens = re.findall(r"[A-Za-z]+|\\[A-Za-z]+|\\d+(?:\\.\\d+)?|[=+*/^<>_-]", s)
    if len(tokens) < cfg.formula_min_tokens:
        return False

    if symbol_ratio >= cfg.formula_min_symbol_ratio and digit_ratio >= cfg.formula_min_digit_ratio:
        return True
    if "=" in s and digit_ratio >= cfg.formula_min_digit_ratio and symbol_ratio >= 0.08:
        return True
    if "/" in s and digit_ratio >= cfg.formula_min_digit_ratio and symbol_ratio >= 0.08:
        return True
    return False


def filter_formula_lines(
    lines: List[str], cfg: ParsePDFConfig, keep_table_blocks: bool
) -> List[str]:
    """Удаляет строки, похожие на математические формулы."""
    if not cfg.remove_formula_lines:
        return lines
    out = []
    for line in lines:
        if keep_table_blocks and "[TABLE" in line:
            out.append(line)
            continue
        if is_formula_line(line, cfg):
            continue
        out.append(line)
    return out


def extract_caption_text_structured(
    caption_block: TextBlock, blocks: List[TextBlock], max_distance: float = 100.0,
    other_table_captions: Optional[List[TextBlock]] = None,
    table_data_blocks: Optional[List[TextBlock]] = None,
) -> str:
    """
    Извлекает полный текст подписи из структурированных блоков.
    
    Ищет блоки, которые являются продолжением подписи (близко расположенные).
    Останавливается при встрече другой таблицы или начала данных таблицы.
    
    Args:
        caption_block: Блок с началом подписи
        blocks: Все блоки страницы
        max_distance: Максимальное расстояние для продолжения подписи
        other_table_captions: Список других подписей таблиц на странице (для предотвращения перекрытий)
        table_data_blocks: Список блоков данных таблицы (для исключения из caption)
    
    Returns:
        Полный текст подписи
    """
    from oasis.parsing.pdf_parser.figures import looks_like_paragraph
    import re
    
    RE_TABLE_CAPTION = re.compile(
        r"^\s*table\s*(s?\d+|[ivxlcdm]+)\s*[\.:]?\s+.*$",
        re.IGNORECASE,
    )
    
    if other_table_captions is None:
        other_table_captions = []
    if table_data_blocks is None:
        table_data_blocks = []
    
    caption_text = caption_block.text
    caption_y = caption_block.y1
    caption_x0 = caption_block.x0
    caption_x1 = caption_block.x1
    
    # ОПТИМИЗАЦИЯ: фильтруем и сортируем блоки заранее
    # Берем только текстовые блоки, которые находятся ниже подписи и в разумных пределах
    candidate_blocks = []
    other_caption_ids = {id(cap) for cap in other_table_captions}
    table_data_block_ids = {id(b) for b in table_data_blocks}  # Исключаем блоки данных таблицы
    
    # Создаем множество координат и текстовых сигнатур блоков данных для проверки
    table_data_bboxes = []
    table_data_text_signatures = set()  # Первые 30 символов каждого блока данных
    for data_block in table_data_blocks:
        table_data_bboxes.append((data_block.x0, data_block.y0, data_block.x1, data_block.y1))
        # Создаем текстовую сигнатуру для идентификации блоков с похожим содержимым
        text_sig = data_block.text.strip()[:50].lower()
        if text_sig:
            table_data_text_signatures.add(text_sig)
    
    def is_table_data_block(block: TextBlock) -> bool:
        """Проверяет, является ли блок частью данных таблицы."""
        if id(block) in table_data_block_ids:
            return True
        # Проверка по координатам (с небольшой погрешностью)
        for x0, y0, x1, y1 in table_data_bboxes:
            if (abs(block.x0 - x0) < 5 and abs(block.y0 - y0) < 5 and 
                abs(block.x1 - x1) < 5 and abs(block.y1 - y1) < 5):
                return True
        # Проверка по текстовой сигнатуре (первые 50 символов)
        block_text_sig = block.text.strip()[:50].lower()
        for sig in table_data_text_signatures:
            # Если блок содержит значительную часть текста из данных таблицы
            if len(sig) > 20 and sig in block_text_sig:
                return True
            if len(block_text_sig) > 20 and block_text_sig in sig:
                return True
        return False
    
    for block in blocks:
        if block.block_type != 0:
            continue
        if block == caption_block:
            continue
        if id(block) in other_caption_ids:
            continue
        if is_table_data_block(block):
            continue  # Пропускаем блоки данных таблицы
        
        vertical_gap = block.y0 - caption_y
        if vertical_gap > 0 and vertical_gap <= max_distance * 2:  # Увеличиваем зону поиска
            candidate_blocks.append(block)
    
    # Сортируем по вертикальной позиции
    candidate_blocks.sort(key=lambda b: b.y0)
    
    # Ищем блоки, которые являются продолжением подписи
    continuation_blocks = []
    
    for block in candidate_blocks:
        vertical_gap = block.y0 - caption_y
        
        if vertical_gap > max_distance:
            break  # Слишком далеко, прекращаем поиск
        
        # Проверяем, не начинается ли блок с подписи другой таблицы
        block_text = block.text.strip()
        if RE_TABLE_CAPTION.match(block_text):
            break
        
        horizontal_overlap = not (block.x1 < caption_x0 or block.x0 > caption_x1)
        
        if horizontal_overlap:
            # Проверяем, не является ли это началом нового параграфа
            if looks_like_paragraph(block.text):
                break
            
            # Проверяем, не похоже ли это на начало данных таблицы
            block_text = block.text.strip()
            if len(block_text) > 0:
                # УНИВЕРСАЛЬНАЯ ПРОВЕРКА: если блок содержит паттерны данных таблицы, не включаем его в caption
                # Паттерны: много чисел подряд, заголовки колонок (короткие слова), диапазоны чисел
                words = block_text.split()
                has_number_ranges = bool(re.search(r'\d+-\d+', block_text))  # Например, "0-2", "3-9"
                has_many_numbers = sum(1 for w in words if re.match(r'^\d+\.?\d*$', w)) >= 3  # 3+ числа
                has_many_ranges = len(re.findall(r'\d+-\d+', block_text)) >= 3  # 3+ диапазона
                has_table_headers = any(kw in block_text.lower() for kw in ['label set', 'default label'])
                
                # СТРОГАЯ ПРОВЕРКА: если блок содержит много диапазонов чисел - это данные таблицы
                # (даже если там есть слова типа "category", это не caption, а данные)
                if has_many_ranges:
                    break  # Это данные таблицы, не caption
                
                # Если блок содержит диапазоны чисел И много чисел - это данные таблицы
                if has_number_ranges and has_many_numbers:
                    break  # Это данные таблицы, не caption
                
                # Если блок содержит заголовки таблиц И много чисел - это данные таблицы
                if has_table_headers and has_many_numbers:
                    break  # Это данные таблицы, не caption
                # 1. Проверка на высокое содержание цифр с разделителями
                digit_ratio = sum(c.isdigit() for c in block_text) / len(block_text)
                has_separators = '|' in block_text or ';' in block_text or '\t' in block_text
                if digit_ratio > 0.3 and has_separators:
                    break
                
                # 2. Проверка на заголовки таблиц: много коротких слов подряд
                # (например, "Middle Southeast East Model Gender Black White Indian")
                words = block_text.split()
                if len(words) >= 5:
                    # Если много коротких слов (заголовки колонок) или есть числа
                    short_words = sum(1 for w in words if len(w) <= 8)
                    has_numbers = any(c.isdigit() for c in block_text)
                    # Если более 60% коротких слов ИЛИ есть числа - это начало данных таблицы
                    if (short_words / len(words) > 0.6) or has_numbers:
                        # Но если это продолжение caption (содержит слова типа "category", "classification")
                        caption_keywords = ['category', 'classification', 'accuracy', 'images', 'fairface', 'percent']
                        if not any(kw in block_text.lower() for kw in caption_keywords):
                            break  # Это начало данных таблицы, не caption
                
                # 3. Проверка на паттерн: заголовки колонок обычно содержат слова типа "Model", "Gender", "Race", "Age"
                # и не содержат длинных предложений
                table_header_keywords = ['model', 'gender', 'race', 'age', 'black', 'white', 'indian', 'latino', 
                                        'eastern', 'asian', 'southeast', 'middle', 'average']
                if any(kw in block_text.lower() for kw in table_header_keywords):
                    # Если это короткий блок с ключевыми словами и нет знаков препинания в конце предложения
                    if len(block_text) < 200 and not block_text.rstrip().endswith(('.', '!', '?', ')')):
                        # Проверяем, не является ли это частью caption (должно быть в конце предложения)
                        if not caption_text.rstrip().endswith(('.', '!', '?', ')')):
                            # Если предыдущий текст не заканчивается на точку, это может быть продолжение caption
                            # Но если блок содержит только ключевые слова таблицы без контекста - это заголовки
                            if len([w for w in words if w.lower() in table_header_keywords]) >= 3:
                                break  # Это заголовки колонок, не caption
            
            # Добавляем к подписи
            if block.text.strip():
                # Если уже есть 2+ предложения и следующий блок похож на параграф, останавливаемся
                text_wo_prefix = re.sub(
                    r"^\s*table\s*(s?\d+|[ivxlcdm]+)\s*[\.:]?\s+",
                    "",
                    caption_text,
                    flags=re.IGNORECASE,
                )
                sentence_count = len(re.findall(r"[.!?]", text_wo_prefix))
                if sentence_count >= 2 and looks_like_paragraph(block.text):
                    break
                # Если подпись одно-строчная, а следующий блок похож на параграф, не продолжаем
                if "\n" not in caption_block.text and looks_like_paragraph(block.text):
                    break
                continuation_blocks.append(block)
                caption_text += " " + block.text
                caption_y = block.y1
    
    # Проверяем, заканчивается ли подпись точкой
    # Если нет, ищем дальше до точки, но с ограничениями
    continuation_block_ids = {id(b) for b in continuation_blocks}
    
    if not caption_text.strip().endswith('.'):
        # Продолжаем поиск только среди оставшихся кандидатов
        for block in candidate_blocks:
            if id(block) in continuation_block_ids:
                continue
            
            block_text = block.text.strip()
            if RE_TABLE_CAPTION.match(block_text):
                break
            
            vertical_gap = block.y0 - caption_y
            if vertical_gap <= 0 or vertical_gap > max_distance:
                continue
            
            horizontal_overlap = not (block.x1 < caption_x0 or block.x0 > caption_x1)
            
            if horizontal_overlap and block.text.strip():
                block_text = block.text.strip()
                
                # УНИВЕРСАЛЬНАЯ ПРОВЕРКА: паттерны данных таблицы (те же, что и в первом цикле)
                words = block_text.split()
                has_number_ranges = bool(re.search(r'\d+-\d+', block_text))  # Например, "0-2", "3-9"
                has_many_numbers = sum(1 for w in words if re.match(r'^\d+\.?\d*$', w)) >= 3  # 3+ числа
                has_many_ranges = len(re.findall(r'\d+-\d+', block_text)) >= 3  # 3+ диапазона
                has_table_headers = any(kw in block_text.lower() for kw in ['label set', 'default label'])
                
                # СТРОГАЯ ПРОВЕРКА: если блок содержит много диапазонов чисел - это данные таблицы
                if has_many_ranges:
                    break  # Это данные таблицы, не caption
                
                # Если блок содержит диапазоны чисел И много чисел - это данные таблицы
                if has_number_ranges and has_many_numbers:
                    break  # Это данные таблицы, не caption
                
                # Если блок содержит заголовки таблиц И много чисел - это данные таблицы
                if has_table_headers and has_many_numbers:
                    break  # Это данные таблицы, не caption
                
                # Те же проверки, что и выше
                digit_ratio = sum(c.isdigit() for c in block_text) / len(block_text) if len(block_text) > 0 else 0
                has_separators = '|' in block_text or ';' in block_text or '\t' in block_text
                if digit_ratio > 0.3 and has_separators:
                    break
                
                # Проверка на заголовки таблиц
                words = block_text.split()
                if len(words) >= 5:
                    short_words = sum(1 for w in words if len(w) <= 8)
                    has_numbers = any(c.isdigit() for c in block_text)
                    if (short_words / len(words) > 0.6) or has_numbers:
                        caption_keywords = ['category', 'classification', 'accuracy', 'images', 'fairface', 'percent']
                        if not any(kw in block_text.lower() for kw in caption_keywords):
                            break
                
                # Проверка на ключевые слова таблиц
                table_header_keywords = ['model', 'gender', 'race', 'age', 'black', 'white', 'indian', 'latino', 
                                        'eastern', 'asian', 'southeast', 'middle', 'average']
                if any(kw in block_text.lower() for kw in table_header_keywords):
                    if len(block_text) < 200 and not block_text.rstrip().endswith(('.', '!', '?', ')')):
                        if not caption_text.rstrip().endswith(('.', '!', '?', ')')):
                            if len([w for w in words if w.lower() in table_header_keywords]) >= 3:
                                break
                
                caption_text += " " + block.text
                caption_y = block.y1
                if block.text.strip().endswith('.'):
                    break
                if looks_like_paragraph(block.text):
                    break
    
    # Ограничиваем подпись, если в нее попал параграф
    caption_text = caption_text.strip()
    if caption_text:
        prefix_match = re.match(
            r"^\s*table\s*(s?\d+|[ivxlcdm]+)\s*[\.:]?\s+",
            caption_text,
            flags=re.IGNORECASE,
        )
        search_start = prefix_match.end() if prefix_match else 0
        first_sentence_end = re.search(r"[.!?]\s+", caption_text[search_start:])
        if first_sentence_end:
            end_pos = search_start + first_sentence_end.end()
            remainder = caption_text[end_pos:].strip()
            remainder_flat = remainder.replace("\n", " ").strip()
            if remainder_flat and len(remainder_flat) > 120:
                if re.match(r"^(We|This|These|Our|In|To|It|ZS|Additionally)\b", remainder_flat):
                    caption_text = caption_text[:end_pos].strip()
    
    return caption_text


def parse_pdf_two_views_structured(
    pdf_path: str | Path,
    out_dir: str | Path,
    cfg: ParsePDFConfig,
    trace_id: str | None = None,
) -> Tuple[Path, Path]:
    """
    Парсит PDF в два представления используя структурированный подход.
    
    Использует get_text("dict") для получения блоков с координатами и метаданными.
    Это позволяет лучше определять структуру документа и отделять таблицы от текста.
    
    Args:
        pdf_path: Путь к PDF файлу
        out_dir: Директория для сохранения результатов
        cfg: Конфигурация парсинга
        trace_id: Идентификатор трейса для логирования
    
    Returns:
        Tuple[Path, Path]: Пути к созданным файлам (topics, rag)
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    remove_keys: Set[str] = set()
    if cfg.remove_running_headers_footers:
        remove_keys = collect_running_headers_footers(
            doc,
            top_margin_pt=cfg.top_margin_pt,
            bottom_margin_pt=cfg.bottom_margin_pt,
            min_pages_frac=cfg.min_pages_frac,
        )

    topics_pages: List[str] = []
    rag_pages: List[str] = []

    stop_all = False

    for pi in range(doc.page_count):
        if stop_all:
            break

        page_no = pi + 1
        page = doc.load_page(pi)
        page_height = page.rect.height

        # Извлекаем структурированные блоки
        blocks = extract_structured_blocks(page)

        # Удаляем headers/footers по координатам
        if remove_keys:
            blocks = filter_blocks_by_coordinates(
                blocks, remove_keys, cfg.top_margin_pt, cfg.bottom_margin_pt, page_height
            )

        # Преобразуем блоки в строки для дальнейшей обработки
        # Сортируем блоки сверху вниз
        blocks_sorted = sorted(blocks, key=lambda b: b.y0)
        lines = [b.text for b in blocks_sorted if b.block_type == 0]

        # Обрезаем References
        if cfg.stop_at_references:
            lines, found_refs = cut_references_section(lines)
            if found_refs:
                stop_all = True

        # Подготавливаем два представления
        lines_topics = list(lines)
        lines_rag = list(lines)

        # --- TABLES: используем структурированный подход
        # Извлекаем таблицы из структурированных блоков
        table_blocks_structured = extract_table_blocks_structured(blocks_sorted, page_height)
        # Обрабатываем таблицы
        # ВАЖНО: обрабатываем в обратном порядке (снизу вверх), чтобы индексы не сдвигались
        # при замене предыдущих таблиц
        index_shift_topics = 0
        index_shift_rag = 0

        # Сначала находим индексы всех таблиц и сортируем их в обратном порядке
        table_blocks_with_indices = []
        # Собираем все подписи таблиц для передачи в extract_caption_text_structured
        all_caption_blocks = [cb for cb, _ in table_blocks_structured]
        
        for idx, (caption_block, data_blocks) in enumerate(table_blocks_structured):
            # Извлекаем полный текст подписи, передавая другие подписи и блоки данных для предотвращения перекрытий
            other_captions = [cb for cb in all_caption_blocks if cb != caption_block]
            caption_text = extract_caption_text_structured(
                caption_block, blocks_sorted, 
                other_table_captions=other_captions,
                table_data_blocks=data_blocks  # Передаем блоки данных таблицы для исключения из caption
            )
            
            # Находим индекс подписи в списке строк
            caption_line_idx = None
            # Сначала пробуем точное совпадение (первые 50 символов)
            caption_start_text = caption_block.text.strip()[:50] if caption_block.text.strip() else ""
            for i, line in enumerate(lines):
                if caption_start_text and caption_start_text in line:
                    caption_line_idx = i
                    break
            
            # Если не нашли, пробуем поиск по началу подписи (первое слово + "table")
            if caption_line_idx is None:
                caption_start = caption_block.text.strip().split()[0] if caption_block.text.strip() else ""
                if caption_start:
                    for i, line in enumerate(lines):
                        if caption_start.lower() in line.lower() and "table" in line.lower():
                            caption_line_idx = i
                            break
            
            if caption_line_idx is not None:
                table_blocks_with_indices.append((caption_line_idx, caption_block, data_blocks, caption_text))
        
        # Сортируем по индексу в обратном порядке (снизу вверх)
        table_blocks_with_indices.sort(key=lambda x: x[0], reverse=True)
        
        # Определяем концы подписей для всех таблиц заранее, чтобы избежать перекрытий
        table_ranges = []
        for caption_line_idx, caption_block, data_blocks, caption_text in table_blocks_with_indices:
            from oasis.parsing.pdf_parser.figures import caption_block_end
            cap_end = caption_block_end(lines, caption_line_idx, max_lines=15)
            
            # Проверяем перекрытие с уже обработанными таблицами
            # Если текущая таблица перекрывается с предыдущей, ограничиваем её конец
            if table_ranges:
                prev_start, prev_end = table_ranges[-1]
                # Если текущая таблица начинается внутри диапазона предыдущей, ограничиваем её конец
                if caption_line_idx < prev_end:
                    # Ограничиваем конец текущей таблицы началом предыдущей
                    cap_end = min(cap_end, prev_start - 1)
            
            table_ranges.append((caption_line_idx, cap_end))
        
        # Обновляем table_blocks_with_indices с правильными диапазонами
        table_blocks_with_indices_and_ranges = [
            (caption_line_idx, cap_end, caption_block, data_blocks, caption_text)
            for (caption_line_idx, caption_block, data_blocks, caption_text), (_, cap_end)
            in zip(table_blocks_with_indices, table_ranges)
        ]

        for caption_line_idx, cap_end, caption_block, data_blocks, caption_text in table_blocks_with_indices_and_ranges:
            
            # Используем структурированный текст подписи как финальный caption
            caption = caption_text

            # Извлекаем номер таблицы
            import re
            m = None
            if caption_block.text:
                for line in caption_block.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    m = RE_TABLE_CAPTION.match(line)
                    if m:
                        break
            if not m:
                caption_first_line = caption_text.strip().splitlines()[0] if caption_text else ""
                m = RE_TABLE_CAPTION.match(caption_first_line)
            if not m:
                continue
            table_num = m.group(1)

            # Собираем данные таблицы из структурированных блоков
            # Объединяем близко расположенные блоки для правильной обработки многострочных заголовков
            merged_data_blocks = merge_nearby_blocks(data_blocks, max_gap=3.0)
            
            table_data_lines = []
            for data_block in merged_data_blocks:
                # Фильтруем блоки, которые явно не являются частью таблицы
                # (например, очень длинные блоки - это параграфы)
                # Но не фильтруем слишком строго - некоторые таблицы могут иметь длинные строки
                if len(data_block.text) > 500:  # Увеличиваем лимит для больших таблиц
                    continue
                table_data_lines.append(data_block.text)

            # Парсим таблицу
            # Если нет данных, создаем таблицу с пустыми данными (только caption)
            if table_data_lines:
                cols, rows = parse_table_from_lines(table_data_lines, caption=caption)
            else:
                # Если нет данных таблицы, создаем пустую структуру
                cols = [""]  # Только метка строки
                rows = []
            
            table_id = f"{cfg.paper_id}_p{page_no}_t{table_num}"
            rag_block = format_table_rag(table_id, caption, cols, rows)

            # Заменяем блок таблицы в строках
            # В topics оставляем только подпись
            adjusted_cap_start = caption_line_idx + index_shift_topics
            adjusted_cap_end = cap_end + index_shift_topics
            old_length = adjusted_cap_end - adjusted_cap_start + 1
            lines_topics[adjusted_cap_start : adjusted_cap_end + 1] = [caption]
            new_length = 1
            index_shift_topics += new_length - old_length

            # В rag вставляем блок таблицы
            adjusted_cap_start_rag = caption_line_idx + index_shift_rag
            adjusted_cap_end_rag = cap_end + index_shift_rag
            old_length_rag = adjusted_cap_end_rag - adjusted_cap_start_rag + 1
            lines_rag[adjusted_cap_start_rag : adjusted_cap_end_rag + 1] = [rag_block]
            new_length_rag = 1
            index_shift_rag += new_length_rag - old_length_rag

        # --- FIGURES: обрабатываем фигуры (используем существующую логику)
        from oasis.parsing.pdf_parser.figures import keep_only_figure_captions

        lines_topics = keep_only_figure_captions(
            lines_topics,
            backward_limit=cfg.fig_noise_backward_limit,
            forward_limit=cfg.fig_noise_forward_limit,
            threshold=cfg.fig_noise_threshold,
        )
        lines_rag = keep_only_figure_captions(
            lines_rag,
            backward_limit=cfg.fig_noise_backward_limit,
            forward_limit=cfg.fig_noise_forward_limit,
            threshold=cfg.fig_noise_threshold,
        )

        # Удаляем строки с формулами, если включено
        lines_topics = filter_formula_lines(lines_topics, cfg, keep_table_blocks=False)
        if cfg.remove_formula_lines_in_rag:
            lines_rag = filter_formula_lines(lines_rag, cfg, keep_table_blocks=True)

        topics_pages.append("\n".join(lines_topics).strip())
        rag_pages.append("\n".join(lines_rag).strip())
        
    doc.close()

    topics_text = "\n\n".join([p for p in topics_pages if p])
    rag_text = "\n\n".join([p for p in rag_pages if p])

    # Постобработка: склейка переносов и разворачивание строк в абзацах
    topics_text = postprocess_text(
        topics_text, keep_table_blocks=False, unwrap_lines=cfg.unwrap_lines_in_paragraphs
    )
    rag_text = postprocess_text(
        rag_text, keep_table_blocks=True, unwrap_lines=cfg.unwrap_lines_in_paragraphs
    )

    topics_out = out_dir / f"{cfg.paper_id}_topics.txt"
    rag_out = out_dir / f"{cfg.paper_id}_rag.txt"
    topics_out.write_text(topics_text, encoding="utf-8")
    rag_out.write_text(rag_text, encoding="utf-8")

    return topics_out, rag_out
