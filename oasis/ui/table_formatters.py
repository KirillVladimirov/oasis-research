"""Функции форматирования данных для отображения в Streamlit UI."""

import ast
import html
import json
from typing import Any, Callable

import pandas as pd


def _parse_authors(val: Any) -> list[str]:
    """Парсит авторов из различных форматов.
    
    Args:
        val: Может быть list, Python-список в строке, JSON строка, строка с разделителями
        
    Returns:
        Список авторов
    """
    if val is None:
        return []
    
    # Если уже список
    if isinstance(val, list):
        return [str(a).strip() for a in val if a and str(a).strip()]
    
    # Проверяем NaN
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in {"nan", "none", "null", "[]", ""}:
        return []
    
    # Пробуем распарсить как Python-список (например: ['Z Wu', 'S Pan', ...])
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            # Используем ast.literal_eval для безопасного парсинга Python-литералов
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if a and str(a).strip()]
        except (ValueError, SyntaxError, TypeError):
            # Если не получилось - пробуем дальше
            pass
    
    # Пробуем распарсить как JSON
    if val_str.startswith("[") or val_str.startswith("{"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if a and str(a).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Пробуем разделить по разделителям
    for sep in [";", "|", ","]:
        if sep in val_str:
            authors = [a.strip() for a in val_str.split(sep) if a.strip()]
            if authors:
                return authors
    
    # Если ничего не подошло - возвращаем как одно имя
    return [val_str] if val_str else []


def _parse_additional_urls(val: Any) -> list[str]:
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


def prepare_table_data(
    df: pd.DataFrame,
    extract_pdf_fn: Callable[[str], str] | None = None,
    is_valid_pdf_fn: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Подготавливает данные для отображения в таблице Streamlit.

    Args:
        df: DataFrame с обогащёнными ссылками
        extract_pdf_fn: Функция для извлечения PDF из additional_urls
        is_valid_pdf_fn: Функция для валидации PDF URL

    Returns:
        Список словарей для отображения в таблице:
        [{"Название": str, "DOI": str, "arXiv ID": str, "Год": str, "Venue": str, "Автор": str, "PDF": str, "ARTICLE": str, "Additional URLs": list}, ...]
    """
    def safe_str(val: Any) -> str:
        """Безопасно преобразует значение в строку, обрабатывая NaN."""
        if val is None:
            return ""
        try:
            import pandas as pd
            if pd.isna(val):
                return ""
        except (TypeError, ValueError, ImportError):
            pass
        if isinstance(val, float):
            try:
                import math
                if math.isnan(val):
                    return ""
            except (TypeError, ValueError, ImportError):
                pass
        result = str(val).strip()
        if result.lower() in {"nan", "none", "null"}:
            return ""
        return result

    table_data = []
    for _, row in df.iterrows():
        title = safe_str(row.get("title", "")) or "Без названия"
        if len(title) > 100:
            title = title[:100] + "..."

        # Извлекаем PDF: сначала из pdf_url, если пуст - из additional_urls
        # ВАЖНО: Не валидируем PDF URL при подготовке таблицы - все валидации уже сделаны на этапе 2
        pdf_url_val = safe_str(row.get("pdf_url", "")) if "pdf_url" in df.columns else ""
        if not pdf_url_val and extract_pdf_fn:
            # Если pdf_url пуст или невалиден - проверяем additional_urls
            add_urls_str = safe_str(row.get("additional_urls", ""))
            if add_urls_str:
                pdf_url_val = extract_pdf_fn(add_urls_str)

        # Парсим additional_urls
        additional_urls_list = _parse_additional_urls(row.get("additional_urls", ""))
        
        # Форматируем год
        year_val = row.get("year")
        if year_val is not None:
            try:
                # Проверяем NaN
                if pd.isna(year_val):
                    year_display = ""
                else:
                    # Пробуем преобразовать в int
                    year_int = int(float(year_val))
                    year_display = str(year_int)
            except (ValueError, TypeError):
                year_display = safe_str(year_val)
        else:
            year_display = ""
        
        # Форматируем авторов (используем универсальный парсер)
        authors_list = _parse_authors(row.get("authors", ""))
        
        table_data.append(
            {
                "Название": title,
                "DOI": "" if safe_str(row.get("doi", "")) else "",
                "arXiv ID": "" if safe_str(row.get("arxiv_id", "")) else "",
                "Год": year_display,
                "Venue": "" if safe_str(row.get("venue", "")) else "",
                "Автор": authors_list,
                "PDF": pdf_url_val,
                "ARTICLE": safe_str(row.get("article_url", "")) if "article_url" in df.columns else "",
                "Additional URLs": additional_urls_list,
            }
        )
    return table_data


def build_references_html_table(
    rows: list[dict[str, Any]],
    is_valid_pdf_fn: Callable[[str], bool] | None = None,
) -> str:
    """Строит HTML-таблицу для отображения ссылок в Streamlit.

    Args:
        rows: Список словарей (результат prepare_table_data)
        is_valid_pdf_fn: Функция для валидации PDF URL

    Returns:
        HTML строка с таблицей
    """
    ths = ["#", "Название", "DOI", "arXiv ID", "Год", "Venue", "Автор", "PDF", "Additional URLs"]
    css = (
        "<style>table.oasis{width:100%;border-collapse:collapse;table-layout:fixed;}"
        "table.oasis th,table.oasis td{border-bottom:1px solid #333;padding:6px 8px;text-align:left;vertical-align:top}"
        "table.oasis td:first-child{text-align:center;width:50px;min-width:50px;max-width:50px;}"
        "table.oasis td:nth-child(2){white-space:normal;word-wrap:break-word;max-width:400px;}"
        "table.oasis td:last-child{white-space:normal;word-wrap:break-word;max-width:300px;font-size:0.9em;}"
        "table.oasis tr:hover{background:rgba(255,255,255,0.03)}"
        ".copybtn{border:none;background:transparent;cursor:pointer;margin-left:6px}"
        ".copybtn:hover{opacity:.8}"
        ".url-list{list-style:none;margin:0;padding:0}"
        ".url-list li{margin:4px 0}"
        ".url-list a{color:#4a9eff;text-decoration:none}"
        ".url-list a:hover{text-decoration:underline}</style>"
    )
    header = "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in ths) + "</tr></thead>"
    body_parts = ["<tbody>"]
    for idx, rr in enumerate(rows, start=1):
        title_raw = str(rr.get("Название", ""))
        title_html = html.escape(title_raw)
        article_url = str(rr.get("ARTICLE", "")).strip()
        if article_url:
            title_cell = f"<a href='{html.escape(article_url)}' target='_blank'>{title_html}</a>"
        else:
            title_cell = title_html
        doi_cell = html.escape(str(rr.get("DOI", "")))
        arxiv_cell = html.escape(str(rr.get("arXiv ID", "")))
        year_cell = html.escape(str(rr.get("Год", ""))) if rr.get("Год") else ""
        venue_cell = html.escape(str(rr.get("Venue", "")))
        
        # Форматируем авторов как список
        authors_list = rr.get("Автор", [])
        if authors_list and isinstance(authors_list, list) and len(authors_list) > 0:
            authors_items = [f"<li>{html.escape(str(author))}</li>" for author in authors_list if author]
            if authors_items:
                author_cell = f"<ul class='url-list'>{''.join(authors_items)}</ul>"
            else:
                author_cell = ""
        else:
            author_cell = ""
        pdf_url = str(rr.get("PDF", "")).strip()
        # ВАЖНО: Не валидируем PDF URL при отображении таблицы - все валидации уже сделаны на этапе 2
        # Простая проверка формата URL (без HTTP-запросов)
        if pdf_url and pdf_url.lower() not in {"nan", "none", "null", ""}:
            pdf_cell = f"<a href='{html.escape(pdf_url)}' target='_blank'>Открыть</a>"
        else:
            pdf_cell = ""
        
        # Форматируем additional_urls как список кликабельных ссылок
        additional_urls = rr.get("Additional URLs", [])
        if additional_urls and isinstance(additional_urls, list):
            url_items = []
            for url in additional_urls:
                url_str = str(url).strip()
                if url_str and url_str.startswith(("http://", "https://")):
                    # Обрезаем длинные URL для отображения
                    display_url = url_str
                    if len(display_url) > 60:
                        display_url = display_url[:57] + "..."
                    url_items.append(
                        f"<li><a href='{html.escape(url_str)}' target='_blank' title='{html.escape(url_str)}'>{html.escape(display_url)}</a></li>"
                    )
            if url_items:
                additional_cell = f"<ul class='url-list'>{''.join(url_items)}</ul>"
            else:
                additional_cell = ""
        else:
            additional_cell = ""
        
        body_parts.append(
            "<tr>"
            + f"<td>{idx}</td><td>{title_cell}</td><td>{doi_cell}</td><td>{arxiv_cell}</td><td>{year_cell}</td><td>{venue_cell}</td><td>{author_cell}</td><td>{pdf_cell}</td><td>{additional_cell}</td>"
            + "</tr>"
        )
    body_parts.append("</tbody>")
    return css + "<table class='oasis'>" + header + "".join(body_parts) + "</table>"

