#!/usr/bin/env python3
"""
OASIS Streamlit UI Application
Main interface for the Ontology-Augmented Survey Synthesis.
"""

import ast
import uuid
from pathlib import Path
from typing import Any

import csv

import pandas as pd
import streamlit as st

from oasis.enrichment.metrics import calculate_reference_metrics
from oasis.pipelines import enrich_references, extract_reference_titles
from oasis.ui.table_formatters import build_references_html_table, prepare_table_data
from oasis.utils.url import extract_pdf_from_additional_urls
from oasis.utils.logging_config import setup_logging

# ВАЖНО: Все валидации PDF URL выполняются на этапе 2 (enrichment),
# в UI только отображаем уже проверенные данные без дополнительных HTTP-запросов

# Page config
st.set_page_config(
    page_title="OASIS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальная инициализация логирования (один раз при старте)
if "logging_initialized" not in st.session_state:
    setup_logging(log_dir="logs", level="DEBUG")
    st.session_state.logging_initialized = True

# Инициализация trace_id в сессии
st.session_state.setdefault("trace_id", str(uuid.uuid4()))


def _parse_authors_for_display(val: Any) -> str:
    """Парсит авторов из различных форматов и возвращает строку для отображения.
    
    Args:
        val: Может быть list, Python-список в строке, JSON строка, строка с разделителями
        
    Returns:
        Строка с авторами через запятую
    """
    if val is None:
        return ""
    
    # Если уже список
    if isinstance(val, list):
        authors_list = [str(a).strip() for a in val if a and str(a).strip()]
        return ", ".join(authors_list) if authors_list else ""
    
    # Проверяем NaN
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in {"nan", "none", "null", "[]", ""}:
        return ""
    
    # Пробуем распарсить как Python-список (например: ['Z Wu', 'S Pan', ...])
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            # Используем ast.literal_eval для безопасного парсинга Python-литералов
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                authors_list = [str(a).strip() for a in parsed if a and str(a).strip()]
                return ", ".join(authors_list) if authors_list else ""
        except (ValueError, SyntaxError, TypeError):
            # Если не получилось - пробуем дальше
            pass
    
    # Пробуем распарсить как JSON
    try:
        import json
        if val_str.startswith("[") or val_str.startswith("{"):
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                authors_list = [str(a).strip() for a in parsed if a and str(a).strip()]
                return ", ".join(authors_list) if authors_list else ""
    except (json.JSONDecodeError, TypeError, ImportError):
        pass
    
    # Пробуем разделить по разделителям
    for sep in [";", "|", ","]:
        if sep in val_str:
            authors = [a.strip() for a in val_str.split(sep) if a.strip()]
            if authors:
                return ", ".join(authors)
    
    # Если ничего не подошло - возвращаем как есть
    return val_str


def main():
    """Main Streamlit application."""
    st.title(" OASIS - Ontology‑Augmented Survey Synthesis")

    # Вкладки (References второй по счёту)
    tab1, tab_refs, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Корпус", "References", "CQs", "Онтология", "Генерация", "Оценка", "Перевод"]
    )

    # Вкладка "Корпус"
    with tab1:
        st.subheader("Загрузка обзора (PDF)")
        topic = st.text_input("Тема/каталог", "deep_active_learning")
        pdf = st.file_uploader("Загрузите обзор (PDF)", type=["pdf"])
        paper_id = "2405.00334"  # для пилота жёстко; далее извлечём из метаданных

        if st.button("Сохранить PDF") and pdf:
            with st.status("Сохранение файла..."):
                base = Path(f"data/{topic}/{paper_id}")
                base.mkdir(parents=True, exist_ok=True)
                dst = base / "review.pdf"
                dst.write_bytes(pdf.read())
            st.success(f"Сохранено: {dst}")

    # Вкладка "CQs"
    with tab2:
        st.header(" Competency Questions")
        st.info("Модуль генерации competency questions из библиографии")

    # Вкладка "References"
    with tab_refs:
        st.header(" Извлечение библиографических ссылок")

        topic = st.text_input("Тема/каталог", "deep_active_learning", key="refs_topic")
        paper_id = st.text_input(
            "Paper ID", "2405.00334", key="refs_paper_id"
        )  # для пилота жёстко

        pdf_path = Path(f"data/{topic}/{paper_id}/review.pdf")

        col1, col2 = st.columns([1, 3])

        with col1:
            # Кнопки этапов с единым стилем
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                extract_titles_button = st.button(" Стадия 1: Извлечь названия", type="primary", use_container_width=True)
            with col_btn2:
                enrich_button = st.button(" Стадия 2: Обогатить ссылки", type="primary", use_container_width=True)

        with col2:
            if pdf_path.exists():
                st.success(f"PDF найден: {pdf_path}")
            else:
                st.error(f"PDF не найден: {pdf_path}")

        # Проверка наличия CSV файлов
        csv_raw_path = Path(f"data/{topic}/references_raw.csv")
        csv_path = Path(f"data/{topic}/references.csv")

        if csv_raw_path.exists():
            st.info(f" Найден файл стадии 1: {csv_raw_path}")
        if csv_path.exists():
            st.info(f" Найден файл стадии 2: {csv_path}")

        # Стадия 1: Извлечение названий
        if extract_titles_button and pdf_path.exists():
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            def update_progress(stage: str, progress: float, message: str = ""):
                """Обновление прогресса в UI."""
                progress_bar.progress(progress)
                status_text.info(f"**{stage.upper()}**: {message}")

            try:
                csv_path_result, refs = extract_reference_titles(
                    str(pdf_path),
                    topic=topic,
                    trace_id=st.session_state.get("trace_id"),
                    progress_callback=update_progress,
                )

                progress_bar.progress(1.0)
                status_text.success(f" Стадия 1 завершена: найдено {len(refs)} ссылок!")

                # Отображение метрик стадии 1 (все в один ряд)
                with metrics_placeholder.container():
                    st.subheader(" Метрики стадии 1 (извлечение названий)")
                    total = len(refs)
                    with_doi = sum(1 for r in refs if r.get("doi"))
                    with_arxiv_id = sum(1 for r in refs if r.get("arxiv_id"))
                    with_title = sum(1 for r in refs if r.get("title"))
                    with_article_url = sum(1 for r in refs if r.get("article_url"))
                    pct_doi = (with_doi / total * 100) if total > 0 else 0
                    pct_arxiv = (with_arxiv_id / total * 100) if total > 0 else 0
                    pct_article_url = (with_article_url / total * 100) if total > 0 else 0
                    
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Всего ссылок", total)

                    with col2:
                        st.metric("С DOI", f"{with_doi}\n({pct_doi:.1f}%)")

                    with col3:
                        st.metric("С arXiv ID", f"{with_arxiv_id}\n({pct_arxiv:.1f}%)")

                    with col4:
                        st.metric("С названием", with_title)

                    with col5:
                        st.metric("С ссылкой на статью", f"{with_article_url}\n({pct_article_url:.1f}%)")

                # Подготовка данных для таблицы
                table_data = []
                for ref in refs:
                    # Используем title или raw_text как fallback
                    title = ref.get("title", "") or ref.get("raw_text", "")[:100] or "Без названия"
                    article_url = ref.get("article_url", "")
                    
                    # Форматируем год
                    year_val = ref.get("year")
                    if year_val is not None:
                        try:
                            if pd.isna(year_val):
                                year_display = ""
                            else:
                                year_int = int(float(year_val))
                                year_display = str(year_int)
                        except (ValueError, TypeError):
                            year_display = str(year_val) if year_val else ""
                    else:
                        year_display = ""
                    
                    # Форматируем авторов (используем универсальный парсер)
                    authors_display = _parse_authors_for_display(ref.get("authors", ""))
                    
                    table_data.append(
                        {
                            "Название": title[:100] + ("..." if len(title) > 100 else ""),
                            "DOI": "" if ref.get("doi") else "",
                            "arXiv ID": "" if ref.get("arxiv_id") else "",
                            "Год": year_display,
                            "Venue": "" if ref.get("venue") else "",
                            "Автор": authors_display,
                            "Статья": "" if article_url else "",
                        }
                    )

                df_display = pd.DataFrame(table_data)

                # Отображение таблицы
                st.subheader(" Таблица результатов")
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

                # Кнопка скачивания CSV
                csv_data = pd.read_csv(csv_path_result)
                # Убеждаемся что данные нормализованы перед скачиванием
                for col in csv_data.columns:
                    if csv_data[col].dtype == 'object':  # Текстовые колонки
                        csv_data[col] = csv_data[col].astype(str).str.replace("\n", " ").str.replace("\r", " ").str.replace("\t", " ")
                st.download_button(
                    label=" Скачать references_raw.csv",
                    data=csv_data.to_csv(index=False, quoting=csv.QUOTE_ALL).encode("utf-8"),
                    file_name="references_raw.csv",
                    mime="text/csv",
                )

            except Exception as e:
                progress_bar.empty()
                status_text.error(f" Ошибка при извлечении ссылок: {e}")
                st.exception(e)

        # Стадия 2: Обогащение ссылок
        if enrich_button and csv_raw_path.exists():
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            def update_progress_enrich(stage: str, progress: float, message: str = ""):
                """Обновление прогресса в UI для стадии 2."""
                progress_bar.progress(progress)
                status_text.info(f"**{stage.upper()}**: {message}")

            try:
                # Загружаем сырые ссылки
                df_raw = pd.read_csv(csv_raw_path)
                raw_refs = df_raw.to_dict("records")

                csv_path_result, enriched_refs, metrics = enrich_references(
                    raw_refs,
                    topic=topic,
                    trace_id=st.session_state.get("trace_id"),
                    progress_callback=update_progress_enrich,
                )

                progress_bar.progress(1.0)
                status_text.success(f" Стадия 2 завершена: обогащено {metrics['total']} ссылок!")

                # Перечитываем сохранённый CSV, чтобы UI всегда показывал фактические результаты
                df_enriched = pd.read_csv(csv_path_result)
                
                # Метрики по CSV (источник истины) - без валидации, все уже проверено на этапе 2
                metrics_csv = calculate_reference_metrics(
                    df_enriched,
                    extract_pdf_fn=extract_pdf_from_additional_urls,
                    is_valid_pdf_fn=None,
                )
                
                # Диагностика: проверяем реальное количество DOI и arXiv ID в CSV
                doi_count = sum(1 for _, row in df_enriched.iterrows() 
                               if pd.notna(row.get("doi")) and str(row.get("doi", "")).strip().lower() not in {"nan", "none", "null", ""})
                arxiv_count = sum(1 for _, row in df_enriched.iterrows() 
                                 if pd.notna(row.get("arxiv_id")) and str(row.get("arxiv_id", "")).strip().lower() not in {"nan", "none", "null", ""})
                
                # Отображение метрик стадии 2 (все в один ряд)
                with metrics_placeholder.container():
                    st.subheader(" Метрики стадии 2 (обогащение)")
                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                    with col1:
                        st.metric("Всего ссылок", metrics_csv["total"])

                    with col2:
                        pct_doi = metrics_csv["pct_doi"]
                        st.metric("С DOI", f"{metrics_csv['with_doi']}\n({pct_doi:.1f}%)")

                    with col3:
                        pct_arxiv = metrics_csv["pct_arxiv"]
                        st.metric("С arXiv ID", f"{metrics_csv['with_arxiv_id']}\n({pct_arxiv:.1f}%)")

                    with col4:
                        st.metric("С названием", metrics_csv["with_title"])

                    with col5:
                        st.metric("С годом", metrics_csv["with_year"])

                    with col6:
                        st.metric("С venue", metrics_csv["with_venue"])

                    with col7:
                        st.metric("С автором", metrics_csv["with_author"])

                    with col8:
                        st.metric("С PDF", metrics_csv.get("with_pdf", 0))

                # Подготовка данных для таблицы (без валидации - все уже проверено на этапе 2)
                try:
                    st.subheader(" Таблица результатов (обогащённые)")
                    table_data = prepare_table_data(
                        df_enriched,
                        extract_pdf_fn=extract_pdf_from_additional_urls,
                        is_valid_pdf_fn=None,  # Не валидируем - все уже проверено на этапе 2
                    )

                    # Отображение таблицы
                    if table_data:
                        html_table = build_references_html_table(table_data, is_valid_pdf_fn=None)
                        st.markdown(html_table, unsafe_allow_html=True)
                        st.info(f" Отображено {len(table_data)} строк в таблице")
                    else:
                        st.warning("️ Таблица пуста: данные не были подготовлены для отображения")
                        st.write(f"DataFrame shape: {df_enriched.shape}")
                        st.write(f"Колонки: {list(df_enriched.columns)}")
                except Exception as table_error:
                    st.error(f" Ошибка при подготовке таблицы: {table_error}")
                    st.exception(table_error)
                    st.write(f"DataFrame shape: {df_enriched.shape if 'df_enriched' in locals() else 'N/A'}")
                    st.write(f"Колонки: {list(df_enriched.columns) if 'df_enriched' in locals() else 'N/A'}")

                # Кнопка скачивания CSV
                csv_data = pd.read_csv(csv_path_result)
                st.download_button(
                    label=" Скачать references.csv",
                    data=csv_data.to_csv(index=False).encode("utf-8"),
                    file_name="references.csv",
                    mime="text/csv",
                )

            except Exception as e:
                progress_bar.empty()
                status_text.error(f" Ошибка при обогащении ссылок: {e}")
                st.exception(e)

        elif enrich_button and not csv_raw_path.exists():
            st.warning("️ Сначала выполните стадию 1 (Извлечь названия)")

        # Отображение существующих результатов стадии 2 (приоритет, если есть обогащённые данные)
        elif csv_path.exists() and not extract_titles_button and not enrich_button:
            try:
                df_existing = pd.read_csv(csv_path)
                st.subheader(" Метрики стадии 2 (обогащённые данные)")

                # Метрики по существующим данным (без валидации - все уже проверено)
                metrics_existing = calculate_reference_metrics(
                    df_existing,
                    extract_pdf_fn=extract_pdf_from_additional_urls,
                    is_valid_pdf_fn=None,
                )

                # Все метрики в один ряд
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                with col1:
                    st.metric("Всего ссылок", metrics_existing["total"])
                with col2:
                    pct_doi = metrics_existing["pct_doi"]
                    st.metric("С DOI", f"{metrics_existing['with_doi']}\n({pct_doi:.1f}%)")
                with col3:
                    pct_arxiv = metrics_existing["pct_arxiv"]
                    st.metric("С arXiv ID", f"{metrics_existing['with_arxiv_id']}\n({pct_arxiv:.1f}%)")
                with col4:
                    st.metric("С названием", metrics_existing["with_title"])
                with col5:
                    st.metric("С годом", metrics_existing["with_year"])
                with col6:
                    st.metric("С venue", metrics_existing["with_venue"])
                with col7:
                    st.metric("С автором", metrics_existing["with_author"])
                with col8:
                    st.metric("С PDF", metrics_existing.get("with_pdf", 0))

                # Таблица с результатами (без валидации - все уже проверено на этапе 2)
                table_data = prepare_table_data(
                    df_existing,
                    extract_pdf_fn=extract_pdf_from_additional_urls,
                    is_valid_pdf_fn=None,  # Не валидируем - все уже проверено на этапе 2
                )

                st.subheader(" Таблица результатов (обогащённые)")
                st.markdown(
                    build_references_html_table(table_data, is_valid_pdf_fn=None),
                    unsafe_allow_html=True,
                )

                st.download_button(
                    label=" Скачать references.csv",
                    data=df_existing.to_csv(index=False).encode("utf-8"),
                    file_name="references.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Ошибка при загрузке существующих данных: {e}")
                st.exception(e)

        # Отображение существующих результатов стадии 1 (fallback, если нет стадии 2)
        elif csv_raw_path.exists() and not extract_titles_button and not enrich_button:
            try:
                df_raw = pd.read_csv(csv_raw_path)
                st.subheader(" Данные стадии 1 (references_raw.csv)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего ссылок", len(df_raw))
                with col2:
                    with_title = df_raw["title"].notna().sum()
                    st.metric("С названием", with_title)
                with col3:
                    with_id = (df_raw["doi"].notna() | df_raw["arxiv_id"].notna()).sum()
                    st.metric("С DOI/arXiv ID", with_id)

                # Таблица
                table_data = []
                for _, row in df_raw.iterrows():
                    title = str(row.get("title", "")) or str(row.get("raw_text", ""))[:100] or "Без названия"
                    
                    # Форматируем год
                    year_val = row.get("year")
                    if year_val is not None:
                        try:
                            if pd.isna(year_val):
                                year_display = ""
                            else:
                                year_int = int(float(year_val))
                                year_display = str(year_int)
                        except (ValueError, TypeError):
                            year_display = str(year_val) if year_val else ""
                    else:
                        year_display = ""
                    
                    # Форматируем авторов (используем универсальный парсер)
                    authors_display = _parse_authors_for_display(row.get("authors", ""))
                    
                    table_data.append(
                        {
                            "Название": title[:100] + ("..." if len(title) > 100 else ""),
                            "DOI": "" if pd.notna(row.get("doi")) and row.get("doi") else "",
                            "arXiv ID": "" if pd.notna(row.get("arxiv_id")) and row.get("arxiv_id") else "",
                            "Год": year_display,
                            "Venue": "" if pd.notna(row.get("venue")) and row.get("venue") else "",
                            "Автор": authors_display,
                        }
                    )

                df_display = pd.DataFrame(table_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)
                st.download_button(
                    label=" Скачать references_raw.csv",
                    data=df_raw.to_csv(index=False, quoting=csv.QUOTE_ALL).encode("utf-8"),
                    file_name="references_raw.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Ошибка при загрузке данных стадии 1: {e}")

    # Вкладка "Онтология"
    with tab3:
        st.header("️ Ontology")
        st.info("Модуль построения и валидации онтологий")

    # Вкладка "Генерация"
    with tab4:
        st.header("️ Генерация обзоров")
        st.info(
            "Модуль генерации обзоров (E1: RAG-only, E2: ontology-augmented)"
        )

    # Вкладка "Оценка"
    with tab5:
        st.header(" Оценка")
        st.info("Модуль оценки качества обзоров и метрик")

    # Вкладка "Перевод"
    with tab6:
        st.header(" Перевод")
        st.info("Модуль перевода обзоров на русский язык")


if __name__ == "__main__":
    main()
