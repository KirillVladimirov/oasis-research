import re
from typing import List, Optional, Tuple

from oasis.parsing.pdf_parser.figures import (
    caption_block_end,
    is_numeric_line,
    looks_like_paragraph,
)

RE_TABLE_CAPTION = re.compile(
    r"^\s*table\s*(s?\d+|[ivxlcdm]+)\s*[\.:]?\s+.*$",
    re.IGNORECASE,
)


def sanitize_col_name(s: str) -> str:
    """Нормализует имя колонки в snake_case."""
    s = (s or "").strip().lower()
    if not s:
        return ""  # Пустая строка для метки строки остается пустой
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def infer_values_per_row(table_lines: List[str]) -> Optional[int]:
    """
    Определяет количество числовых значений на строку таблицы.

    Args:
        table_lines: Список строк таблицы

    Returns:
        Количество значений на строку или None
    """
    candidates = []
    tl = [x.strip() for x in table_lines if x.strip()]
    
    # Сначала пробуем стандартный подход (по строкам)
    for i in range(len(tl) - 2):
        if (not is_numeric_line(tl[i])) and is_numeric_line(tl[i + 1]):
            run = 0
            j = i + 1
            while j < len(tl) and is_numeric_line(tl[j]):
                run += 1
                j += 1
            if run >= 2:
                candidates.append(run)
    
    # Если не нашли по строкам, пробуем разбить длинные строки на слова
    if not candidates:
        for line in tl:
            words = line.split()
            # Ищем паттерн: не-число + несколько чисел подряд
            for i in range(len(words) - 2):
                if (not is_numeric_line(words[i])) and is_numeric_line(words[i + 1]):
                    run = 0
                    j = i + 1
                    while j < len(words) and is_numeric_line(words[j]):
                        run += 1
                        j += 1
                    if run >= 2:
                        candidates.append(run)
    
    if not candidates:
        return None
    return max(set(candidates), key=candidates.count)


def _merge_header_words(words: List[str], target_count: int) -> List[str]:
    """
    Склеивает слова заголовков в устойчивые пары (например, Middle Eastern),
    чтобы приблизить количество колонок к target_count.
    """
    if len(words) <= target_count:
        return words

    modifiers = {
        "middle", "south", "southeast", "north", "east", "west", "central", "latin", "sub"
    }
    suffixes = {
        "eastern", "asian", "african", "american", "european", "islander", "latino", "hispanic"
    }

    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            w1 = words[i].lower()
            w2 = words[i + 1].lower()
            if w1 in modifiers and w2 in suffixes:
                merged.append(f"{words[i]} {words[i + 1]}")
                i += 2
                continue
        merged.append(words[i])
        i += 1

    while len(merged) > target_count and len(merged) >= 2:
        merged[-2] = f"{merged[-2]} {merged[-1]}"
        merged.pop()

    return merged


def parse_table_from_lines(
    table_lines: List[str], caption: str
) -> Tuple[List[str], List[List[str]]]:
    """
    Парсит таблицу из линейного списка строк в структуру (колонки, строки).

    Args:
        table_lines: Список строк таблицы
        caption: Подпись таблицы

    Returns:
        Tuple[List[str], List[List[str]]]: (колонки, строки)
    """
    tl = []
    for line in table_lines:
        if not line:
            continue
        for part in str(line).splitlines():
            part = part.strip()
            if part:
                tl.append(part)

    values_per_row = infer_values_per_row(tl)
    if values_per_row is None:
        # Резервный вариант: деление по нескольким пробелам
        rows = []
        for l in tl:
            parts = re.split(r"\s{2,}", l.strip())
            if len(parts) >= 2:
                rows.append(parts)
        if not rows:
            return (["col1"], [[" ".join(tl[:30])]])
        n_cols = max(len(r) for r in rows)
        cols = [f"col{i+1}" for i in range(n_cols)]
        norm_rows = [r + [""] * (n_cols - len(r)) for r in rows]
        return cols, norm_rows

    n_cols = values_per_row + 1

    # Ищем первую строку данных
    # Сначала пробуем стандартный подход (по строкам)
    first_row_idx = None
    for i in range(len(tl) - values_per_row):
        if (not is_numeric_line(tl[i])) and all(
            is_numeric_line(x) for x in tl[i + 1 : i + 1 + values_per_row]
        ):
            first_row_idx = i
            break
    
    # Если не нашли по строкам, пробуем разбить длинные строки на слова
    if first_row_idx is None:
        for line_idx, line in enumerate(tl):
            words = line.split()
            # Ищем паттерн: метка строки (не-число) + значения (числа)
            for i in range(len(words) - values_per_row):
                if (not is_numeric_line(words[i])) and all(
                    is_numeric_line(words[i + j]) for j in range(1, values_per_row + 1)
                ):
                    # Нашли первую строку данных
                    first_row_idx = line_idx
                    break
            if first_row_idx is not None:
                break
    
    if first_row_idx is None:
        return (["col1"], [[" ".join(tl[:30])]])

    header_lines = tl[:first_row_idx]
    
    # Специальная обработка для таблиц 3 и 4: Model | Race | Gender | Age
    # Эти таблицы имеют структуру с подзаголовками категорий рас
    # Заголовки могут быть в формате: "Model Gender Black White Indian Latino Middle Eastern Southeast Asian East Asian Average"
    # или разделены на несколько строк
    if len(header_lines) >= 1:
        # Проверяем, содержит ли caption упоминание "Race, Gender, and Age"
        if re.search(r'\b(race|gender|age)\b.*\b(race|gender|age)\b.*\b(race|gender|age)\b', caption, re.IGNORECASE):
            # Это таблица с колонками Model, Race, Gender, Age
            # Ищем ключевые слова в заголовках
            header_text = ' '.join(header_lines).lower()
            if 'model' in header_text and ('race' in header_text or 'gender' in header_text or 'age' in header_text):
                # Пытаемся найти правильные заголовки
                # Ожидаем: Model, Race, Gender, Age (или их вариации)
                found_headers = []
                if 'model' in header_text:
                    found_headers.append('model')
                if 'race' in header_text:
                    found_headers.append('race')
                if 'gender' in header_text:
                    found_headers.append('gender')
                if 'age' in header_text:
                    found_headers.append('age')
                
                # Если нашли все 4 заголовка, используем их
                if len(found_headers) >= 3:  # Может быть без Age в некоторых случаях
                    # Добавляем пустую первую колонку для метки строки
                    header_lines = [''] + [h.capitalize() for h in found_headers]
                    # Если не хватает до n_cols, добавляем недостающие
                    while len(header_lines) < n_cols:
                        header_lines.append(f"col{len(header_lines)}")
    
    # Специальная обработка для случая, когда заголовки находятся в нескольких строках
    # и нужно правильно их распределить по колонкам
    # Например: ['Majority Vote Accuracy on Guesses', 'Accuracy Majority Vote on Full Dataset Accuracy on Guesses']
    # должно стать: ['', 'Accuracy', 'Majority Vote on Full Dataset', 'Accuracy on Guesses', 'Majority Vote Accuracy on Guesses']
    
    # Если у нас 2 строки заголовков и они содержат ключевые слова, пытаемся разобрать их
    if len(header_lines) == 2 and n_cols == 5:
        line1 = header_lines[0]
        line2 = header_lines[1]
        
        # Проверяем, начинается ли вторая строка с 'Accuracy'
        if line2.startswith('Accuracy'):
            # Пытаемся разобрать структуру
            # Вторая строка: 'Accuracy Majority Vote on Full Dataset Accuracy on Guesses'
            # Первая строка: 'Majority Vote Accuracy on Guesses'
            
            # Разбиваем вторую строку на части
            words2 = line2.split()
            # Ищем паттерн: 'Accuracy' ... 'Majority Vote on Full Dataset' ... 'Accuracy on Guesses'
            # Или просто разбиваем по 'Accuracy'
            
            # Простая эвристика: если в строке есть несколько вхождений ключевых слов,
            # пытаемся разбить по ним
            if 'Accuracy' in words2 and words2.count('Accuracy') >= 2:
                # Находим все позиции 'Accuracy'
                acc_positions = [i for i, w in enumerate(words2) if w == 'Accuracy']
                
                # Если есть хотя бы 2 вхождения, пытаемся разбить
                if len(acc_positions) >= 2:
                    # Первая колонка: пустая (метка строки)
                    # Вторая колонка: 'Accuracy' (первое вхождение)
                    # Третья колонка: между первым и вторым 'Accuracy'
                    # Четвертая колонка: 'Accuracy on Guesses' (второе вхождение + следующее)
                    # Пятая колонка: первая строка 'Majority Vote Accuracy on Guesses'
                    
                    col2 = 'Accuracy'
                    col3 = ' '.join(words2[acc_positions[0]+1:acc_positions[1]])
                    col4 = ' '.join(words2[acc_positions[1]:acc_positions[1]+3]) if acc_positions[1]+3 <= len(words2) else ' '.join(words2[acc_positions[1]:])
                    col5 = line1
                    
                    header_lines = ['', col2, col3, col4, col5]
    
    # Улучшенная обработка многострочных заголовков
    # Если заголовков больше, чем колонок, пытаемся правильно их распределить
    # Но только если header_lines еще не был обработан специальной логикой выше
    if len(header_lines) != n_cols and len(header_lines) > n_cols:
        # Специальная обработка для заголовков, разбитых на одно слово в строке
        # (например, "Middle", "Southeast", "East", "Category", "Black", ...).
        if all(" " not in h for h in header_lines) and "Category" in header_lines:
            words = header_lines[:]
            cat_idx = words.index("Category")
            before = words[:cat_idx]
            after = words[cat_idx + 1 :]
            base_races = ["Black", "White", "Indian", "Latino"]
            base_cols = []
            while after and base_races and after[0] == base_races[0]:
                base_cols.append(after.pop(0))
                base_races.pop(0)
            suffixes = after[:]
            merged_tail = []
            for modifier in before:
                mod_lower = modifier.lower()
                if mod_lower == "middle" and "Eastern" in suffixes:
                    suffixes.remove("Eastern")
                    merged_tail.append(f"{modifier} Eastern")
                elif mod_lower in ("southeast", "east") and "Asian" in suffixes:
                    suffixes.remove("Asian")
                    merged_tail.append(f"{modifier} Asian")
                elif suffixes:
                    merged_tail.append(f"{modifier} {suffixes.pop(0)}")
                else:
                    merged_tail.append(modifier)
            header_lines = ["Category"] + base_cols + merged_tail
            if len(header_lines) < n_cols:
                header_lines += [f"col{i+1}" for i in range(len(header_lines), n_cols)]
            header_lines = header_lines[:n_cols]
        else:
            # Пытаемся определить паттерн распределения заголовков
            # Если все заголовки короткие (< 30 символов), возможно, они разбиты на несколько строк
            # Используем эвристику: если заголовков примерно в 2 раза больше колонок,
            # распределяем их по 2 строки на колонку
            all_short = all(len(h) < 30 for h in header_lines)
        
            if all_short and len(header_lines) >= n_cols * 1.5:
                # Распределяем заголовки более умно
                # Первая колонка обычно - метка строки (без заголовка)
                # Остальные колонки - данные с заголовками
                data_cols = n_cols - 1
                headers_per_col = len(header_lines) // data_cols
                remainder = len(header_lines) % data_cols
                
                merged_headers = [""]  # Первая колонка - метка строки, без заголовка
                start_idx = 0
                for col_idx in range(data_cols):
                    # Определяем количество заголовков для текущей колонки данных
                    num_headers = headers_per_col + (1 if col_idx < remainder else 0)
                    end_idx = start_idx + num_headers
                    
                    # Объединяем заголовки для текущей колонки
                    col_headers = header_lines[start_idx:end_idx]
                    merged_header = " ".join(col_headers)
                    merged_headers.append(merged_header)
                    
                    start_idx = end_idx
                
                header_lines = merged_headers
            else:
                # Если заголовки не все короткие или их не так много, используем старую логику
                header_lines = header_lines[: n_cols - 1] + [" ".join(header_lines[n_cols - 1 :])]
    elif len(header_lines) < n_cols:
        # Если заголовков меньше, добавляем пустые
        # Первая колонка - метка строки, может быть без заголовка
        
        # Если заголовков ровно на колонки данных (n_cols - 1), добавляем пустой для метки строки
        if len(header_lines) == n_cols - 1:
            header_lines = [""] + header_lines
        elif len(header_lines) == 1 and n_cols > 1:
            # Один заголовок - это может быть заголовок для всех колонок данных
            # Разбиваем его на отдельные заголовки
            header_text = header_lines[0]
            # Пытаемся разбить по пробелам или другим разделителям
            words = header_text.split()
            if len(words) >= n_cols:
                words = _merge_header_words(words, n_cols)
                header_lines = words[:n_cols]
            elif len(words) >= n_cols - 1:
                # Достаточно слов для всех колонок данных
                words = _merge_header_words(words, n_cols - 1)
                header_lines = [""] + words[:n_cols - 1]
            else:
                # Недостаточно слов - добавляем пустую первую колонку и оставшиеся
                header_lines = [""] + header_lines + [
                    f"col{i+1}" for i in range(len(header_lines), n_cols - 1)
                ]
        else:
            # Меньше заголовков - добавляем пустую первую колонку и оставшиеся
            header_lines = [""] + header_lines + [
                f"col{i+1}" for i in range(len(header_lines), n_cols - 1)
            ]

    cols = [sanitize_col_name(h) for h in header_lines]

    # Если в подписи есть Top-1, добавляем суффикс для числовых колонок
    if re.search(r"\btop-?1\b", caption, re.IGNORECASE):
        cols = [cols[0]] + [
            c if c.endswith("_top1") else f"{c}_top1" for c in cols[1:]
        ]

    # Делаем имена колонок уникальными
    seen = {}
    uniq = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}_{seen[c]}")
    cols = uniq

    rows: List[List[str]] = []
    i = first_row_idx
    
    # Обрабатываем все строки данных
    # Каждая строка может содержать несколько строк таблицы в одной строке текста
    # (например, "Male 96.9 96.4 98.7 96.5 98.9 96.2 96.9 97.2 Linear Probe CLIP Female 97.9...")
    while i < len(tl):
        line = tl[i]
        words = line.split()
        
        # Проверяем, содержит ли строка несколько строк таблицы
        # Паттерн: "метка число число ... метка число ..."
        potential_rows = []
        j = 0
        max_iterations = len(words) * 2  # Защита от бесконечного цикла
        iteration_count = 0
        
        while j < len(words) and iteration_count < max_iterations:
            iteration_count += 1
            # Ищем начало строки таблицы: не-число (метка)
            if not is_numeric_line(words[j]):
                label_parts = [words[j]]
                j += 1
                # Собираем метку (может состоять из нескольких слов)
                while j < len(words) and not is_numeric_line(words[j]):
                    label_parts.append(words[j])
                    j += 1
                
                label = " ".join(label_parts)
                
                # Собираем значения (числа)
                vals = []
                while j < len(words) and is_numeric_line(words[j]) and len(vals) < values_per_row:
                    vals.append(words[j])
                    j += 1
                
                # Если нашли правильное количество значений, это строка таблицы
                if len(vals) == values_per_row:
                    potential_rows.append([label] + vals)
                elif len(vals) > 0 and j >= len(words):
                    # Неполная строка в конце - это может быть последняя строка
                    potential_rows.append([label] + vals)
                    break
            else:
                # Если текущий элемент - число, пропускаем его
                j += 1
        
        # Если нашли несколько строк в разбитом виде, используем их
        if potential_rows and len(potential_rows) >= 1:
            rows.extend(potential_rows)
            i += 1
        else:
            # Используем стандартную логику (одна строка таблицы на строку текста)
            if i + values_per_row < len(tl) and (not is_numeric_line(tl[i])) and all(
                is_numeric_line(x) for x in tl[i + 1 : i + 1 + values_per_row]
            ):
                row_label = tl[i]
                vals = tl[i + 1 : i + 1 + values_per_row]
                rows.append([row_label] + vals)
                i += 1 + values_per_row
            else:
                i += 1

    return cols, rows


def collect_table_block_before_caption(
    lines: List[str], cap_start_idx: int, lookback_lines: int = 220
) -> Tuple[int, List[str]]:
    """
    Собирает блок таблицы перед подписью.
    Останавливается на параграфах и подписях фигур, чтобы не захватывать текст из картинок.

    Args:
        lines: Список строк страницы
        cap_start_idx: Индекс начала подписи таблицы
        lookback_lines: Максимальное количество строк для поиска назад

    Returns:
        Tuple[int, List[str]]: (индекс начала блока, строки блока)
    """
    import re
    from oasis.parsing.pdf_parser.figures import RE_FIG_CAPTION, looks_like_paragraph
    
    block = []
    j = cap_start_idx - 1
    steps = 0
    consecutive_empty = 0
    paragraph_lines_count = 0
    last_empty_idx = None
    
    # Ищем начало таблицы - обычно это короткие строки с заголовками колонок
    # или строки с данными (числа, короткие метки)
    def looks_like_table_start(s: str) -> bool:
        """Проверяет, похожа ли строка на начало таблицы (заголовок или данные)."""
        s = s.strip()
        if not s:
            return False
        # Короткие строки с заголовками (обычно < 50 символов)
        if len(s) < 50:
            # Проверяем, не является ли это параграфом
            if looks_like_paragraph(s):
                return False
            # Проверяем паттерны заголовков таблиц
            # Заголовки часто содержат слова типа "Accuracy", "Vote", "Dataset" и т.д.
            table_keywords = ['accuracy', 'vote', 'dataset', 'guesses', 'majority', 'full']
            s_lower = s.lower()
            if any(keyword in s_lower for keyword in table_keywords):
                return True
            # Или просто короткая строка без пунктуации в конце (кроме двоеточия)
            if not s.endswith(('.', ',', ';')) or s.endswith(':'):
                return True
        # Строки с числами и метками (паттерн таблицы)
        if re.search(r'^\s*[A-Z][a-z]+.*\d+', s) or re.search(r'^\s*\d+', s):
            return True
        return False
    
    while j >= 0 and steps < lookback_lines:
        s = lines[j].strip()
        
        # Останавливаемся на пустых строках (граница параграфа)
        if not s:
            consecutive_empty += 1
            last_empty_idx = j
            if consecutive_empty >= 2:  # Две пустые строки подряд = граница параграфа
                break
            # Если одна пустая строка и уже собрали достаточно строк таблицы
            if len(block) >= 5:
                # Проверяем контекст: если перед пустой строкой идет параграф, останавливаемся
                if j > 0:
                    prev_s = lines[j-1].strip()
                    if prev_s and looks_like_paragraph(prev_s):
                        # Это граница между параграфом и таблицей
                        break
            j -= 1
            steps += 1
            continue
        
        consecutive_empty = 0
        
        # Останавливаемся на подписях фигур (чтобы не захватывать текст из картинок)
        if RE_FIG_CAPTION.match(s):
            break
        
        # Останавливаемся на подписях других таблиц
        if j < cap_start_idx - 1:
            m = RE_TABLE_CAPTION.match(s)
            if m:
                break
        
        # КЛЮЧЕВАЯ ПРОВЕРКА: Проверяем строку, которая идет ПОСЛЕ текущей в исходном порядке (j+1)
        # Если она - начало таблицы, останавливаемся на текущей строке (не включаем её в блок)
        # НО: не останавливаемся слишком рано - нужно собрать хотя бы несколько строк таблицы
        if len(block) > 3 and j + 1 < cap_start_idx:
            next_s_forward = lines[j + 1].strip()
            if next_s_forward and looks_like_table_start(next_s_forward):
                # Следующая строка в исходном порядке - начало таблицы
                # Проверяем, является ли текущая строка параграфом или похожа на конец параграфа
                if looks_like_paragraph(s) or (len(s) > 50 and not looks_like_table_start(s)):
                    # Текущая строка - параграф или не таблица, следующая - начало таблицы
                    # Останавливаемся здесь (не включаем текущую строку)
                    break
        
        # Проверяем, является ли строка частью параграфа
        is_paragraph = looks_like_paragraph(s)
        if is_paragraph:
            paragraph_lines_count += 1
            # Если несколько строк подряд похожи на параграф, это точно параграф
            if paragraph_lines_count >= 2:
                # Останавливаемся - это начало параграфа перед таблицей
                break
            # Если одна строка похожа на параграф и она очень длинная, останавливаемся
            if paragraph_lines_count == 1 and len(s) > 100:
                break
        else:
            # Сброс счетчика параграфов при встрече не-параграфа
            if paragraph_lines_count > 0:
                paragraph_lines_count = 0
        
        block.append(lines[j])
        j -= 1
        steps += 1
    
    start_idx = j + 1
    return start_idx, list(reversed(block))


def format_table_rag(
    table_id: str, caption: str, cols: List[str], rows: List[List[str]]
) -> str:
    """
    Форматирует таблицу в RAG формат.

    Args:
        table_id: Идентификатор таблицы
        caption: Подпись таблицы
        cols: Список колонок
        rows: Список строк (каждая строка - список значений)

    Returns:
        Отформатированная строка в формате [TABLE]...[/TABLE]
    """
    from oasis.parsing.pdf_parser.postprocess import fix_hyphenated_linebreaks
    
    out = []
    out.append(f"[TABLE id={table_id}]")
    # Исправляем переносы слов в caption
    # Переносы могут быть в виде "classi- ﬁcation" (дефис + пробел) или "classiﬁca- tion" (дефис + пробел)
    import re
    caption_fixed = caption.strip()
    # Исправляем переносы с пробелами: "word- word" -> "wordword"
    # Учитываем различные типы дефисов и пробелов
    caption_fixed = re.sub(r"([A-Za-zА-Яа-я])[\-\u2010\u2011\u2012\u2013\u2014]\s+([A-Za-zА-Яа-я])", r"\1\2", caption_fixed)
    # Также применяем стандартную функцию для переносов с переводами строк
    caption_fixed = fix_hyphenated_linebreaks(caption_fixed)
    out.append(f"caption: {caption_fixed}")
    
    # Форматируем колонки: первая колонка может быть пустой (метка строки)
    # Если первая колонка пустая, пропускаем её в выводе columns
    # Но сохраняем её в данных строк
    if cols and not cols[0].strip():
        # Первая колонка пустая - это метка строки, не выводим её в columns
        non_empty_cols = [c for c in cols[1:] if c.strip()]
    else:
        # Все колонки имеют заголовки
        non_empty_cols = [c for c in cols if c.strip()]
    
    out.append("columns: " + " | ".join(non_empty_cols))
    
    for r in rows:
        pairs = []
        row_label = None
        
        # Сначала обрабатываем первую колонку (метка строки)
        if cols and not cols[0].strip() and len(r) > 0:
            # Первая колонка пустая - это метка строки
            row_label = str(r[0]).strip().replace("\n", " ")
            # Обрабатываем остальные колонки
            for idx, (c, v) in enumerate(zip(cols[1:], r[1:]), start=1):
                if c.strip():
                    v2 = str(v).strip().replace("\n", " ")
                    pairs.append(f"{c}={v2}")
        else:
            # Все колонки имеют заголовки
            for idx, (c, v) in enumerate(zip(cols, r)):
                if c.strip():
                    v2 = str(v).strip().replace("\n", " ")
                    pairs.append(f"{c}={v2}")
        
        # Формируем строку: метка строки идет первой без знака "=", затем пары ключ=значение
        if row_label:
            row_str = f"{row_label}; " + "; ".join(pairs) if pairs else row_label
        else:
            row_str = "; ".join(pairs)
        
        out.append("row: " + row_str)
    out.append("[/TABLE]")
    return "\n".join(out)


def count_tables_in_text(text: str) -> int:
    """
    Подсчитывает количество таблиц в тексте RAG версии.

    Использует подсчет открывающих тегов [TABLE id=...], т.к. format_table_rag()
    всегда создает парные теги [TABLE]...[/TABLE]. Учитывает особенности
    форматирования после постобработки (unwrap).

    Args:
        text: Текст RAG версии

    Returns:
        Количество найденных таблиц
    """
    import re

    # Ищем все открывающие теги [TABLE id=...]
    # Учитываем возможные пробелы: [TABLE id=...] или [TABLE  id=...]
    # Используем finditer для более точного подсчета
    opening_pattern = re.compile(r"\[TABLE\s+id=", re.IGNORECASE)
    matches = list(opening_pattern.finditer(text))
    
    return len(matches)


def extract_table_blocks(text: str) -> List[str]:
    """
    Извлекает все блоки таблиц [TABLE]...[/TABLE] из текста.

    Args:
        text: Текст RAG версии

    Returns:
        Список строк с полными блоками таблиц
    """
    import re

    blocks = []
    # Находим все открывающие теги (учитываем возможные пробелы)
    opening_pattern = re.compile(r"\[TABLE\s+id=[^\]]+\]", re.IGNORECASE)
    opening_tags = list(opening_pattern.finditer(text))
    
    closing_pattern = re.compile(r"\[/TABLE\]", re.IGNORECASE)
    
    for match in opening_tags:
        start_pos = match.start()
        remaining_text = text[start_pos:]
        # Ищем закрывающий тег после открывающего
        closing_match = closing_pattern.search(remaining_text)
        if closing_match:
            end_pos = start_pos + closing_match.end()
            block = text[start_pos:end_pos]
            blocks.append(block)
    
    return blocks


def extract_table_ids(text: str) -> List[str]:
    """
    Извлекает все ID таблиц из текста RAG версии.

    Args:
        text: Текст RAG версии

    Returns:
        Список ID таблиц
    """
    import re

    # Учитываем возможные пробелы и варианты написания
    table_ids = re.findall(r"\[TABLE\s+id=([^\]]+)\]", text, re.IGNORECASE)
    return table_ids


def print_tables_as_rag(pdf_path: str, paper_id: str = "paperX") -> None:
    """
    Диагностическая функция: печатает все таблицы из PDF в RAG-формате.
    
    Использует структурированный подход для лучшего определения структуры таблиц.

    Args:
        pdf_path: Путь к PDF файлу
        paper_id: Идентификатор статьи
    """
    from oasis.parsing.pdf_parser.parser_structured import parse_pdf_two_views_structured
    from oasis.parsing.pdf_parser.config import ParsePDFConfig
    from pathlib import Path
    import tempfile
    import os

    pdf_path = Path(pdf_path)
    
    # Используем структурированный парсер для получения правильных результатов
    # Создаем временную директорию для результатов
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ParsePDFConfig(paper_id=paper_id)
        topics_path, rag_path = parse_pdf_two_views_structured(
            pdf_path, tmpdir, config
        )
        
        # Извлекаем все таблицы из RAG версии
        rag_text = rag_path.read_text(encoding="utf-8")
        table_blocks = extract_table_blocks(rag_text)
        
        # Выводим все таблицы
        for block in table_blocks:
            print(block)
            print("\n" + "=" * 80 + "\n")
