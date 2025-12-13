# OASIS

OASIS (Ontology-Augmented Survey Synthesis) - система для генерации научных обзоров с использованием онтологий.

## Repository Structure

OASIS/
 app.py                        # Streamlit UI application
 Makefile                      # convenient make commands for each step
 requirements.txt              # Python dependencies
 setup.py                      # package setup configuration
 docker-compose.yml            # Services: Weaviate, Fuseki, GROBID
 .env.sample                   # sample environment variables (API keys, etc.)
 oasis/                        # библиотечный код
    utils/
        network.py            # сетевые утилиты (проверка здоровья сервисов)
 scripts/                       # CLI скрипты
    check_services_health.py  # проверка здоровья сервисов
 configs/                       # конфигурационные файлы (YAML)
 data/                         # рабочие данные по темам
 ont/                          # онтологические файлы
    shacl/                    # SHACL shapes для валидации
 bib/                          # библиографические данные (BibTeX)
 cqs/                          # competency questions
 docs/                         # документация проекта
    architecture/            # архитектурные заметки
    queries_v1.md            # запросы и промпты
 reports/                      # отчёты и манифесты
    bootstrap/                # манифесты bootstrap (run_manifest.json)
 tests/                        # тесты
 logs/                         # логи приложения
 notes/                        # заметки
 weaviate_data/                # volume для Weaviate (файлы векторного индекса)

## Быстрый старт

### Требования

- **Python**: >= 3.11
- **Docker**: Docker и docker-compose должны быть установлены и запущены
- **Память**: Рекомендуется не менее 8GB RAM (особенно для GROBID)

### Установка и настройка

1. **Клонирование репозитория** (если еще не клонирован):
   ```bash
   git clone <repository-url>
   cd OASIS
   ```

2. **Настройка окружения**:
   ```bash
   # Автоматическая настройка (рекомендуется)
   make bootstrap
   ```
   
   Или вручную:
   ```bash
   # Проверка Python версии
   make check-python
   
   # Создание виртуального окружения (если отсутствует)
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Установка зависимостей
   make setup
   ```

3. **Настройка переменных окружения**:
   ```bash
   # Скопировать шаблон
   cp .env.sample .env
   
   # Отредактировать .env и заполнить:
   # - OPENAI_API_KEY (обязательно)
   # - OPENAI_API_BASE (по умолчанию: https://api.deepseek.com)
   # - LANGFUSE_* (опционально, для трейсинга)
   ```

4. **Запуск Docker-сервисов**:
   ```bash
   # Если Docker использует стандартный socket
   export DOCKER_HOST=unix:///var/run/docker.sock
   
   # Запуск всех сервисов (Weaviate, Fuseki, GROBID)
   make services-up
   ```
   
   Или напрямую:
   ```bash
   docker compose up -d
   ```

5. **Проверка работоспособности**:
   ```bash
   make health-check
   ```
   
   Ожидаемый вывод:
   ```
   Weaviate: 
   Fuseki: 
   GROBID: 
   All services are healthy 
   ```

## Работа с Streamlit UI

### Запуск приложения

После успешной настройки окружения можно запустить веб-интерфейс:

```bash
# Запуск через Makefile (рекомендуется)
make run

# Или напрямую
source .venv/bin/activate
streamlit run app.py
```

Приложение автоматически откроется в браузере по адресу `http://localhost:8501`.

### Загрузка обзора (PDF) — вкладка "Корпус"

Система предоставляет интерфейс для загрузки PDF обзоров через веб-интерфейс:

1. **Откройте вкладку "Корпус"** в главном меню приложения

2. **Заполните параметры**:
   - **Тема/каталог**: название темы исследования (по умолчанию: `deep_active_learning`)
   - **Paper ID**: для пилота используется фиксированный ID `2405.00334` (в будущем будет автоматически извлекаться из метаданных)

3. **Загрузите PDF файл**:
   - Нажмите "Browse files" в загрузчике файлов
   - Выберите PDF файл обзора
   - Нажмите кнопку "Сохранить PDF"

4. **Проверка результата**:
   - После успешного сохранения появится сообщение с путём к файлу
   - Файл будет сохранён в структуре: `data/<topic>/<paper_id>/review.pdf`
   - Например: `data/deep_active_learning/2405.00334/review.pdf`

### Структура сохранения данных

Загруженные PDF файлы сохраняются в следующей структуре:

```
data/
 <topic>/              # название темы (например: deep_active_learning)
     <paper_id>/       # ID статьи (например: 2405.00334)
         review.pdf    # загруженный PDF обзор
```

**Пример**:
```
data/deep_active_learning/2405.00334/review.pdf
```

### Доступные вкладки

Приложение предоставляет следующие вкладки для работы:

- **Корпус**: загрузка PDF обзоров и управление корпусом документов
- **CQs**: генерация competency questions из библиографии
- **Онтология**: построение и валидация онтологий
- **Генерация**: генерация обзоров (E1: RAG-only, E2: ontology-augmented)
- **Оценка**: оценка качества обзоров и метрик
- **Перевод**: перевод обзоров на русский язык

## Порядок запуска пайплайна (онтология и граф знаний)

Ниже приведена типовая последовательность CLI-скриптов, которыми формируется онтология и итоговый граф знаний. Перед запуском каждого шага активируйте виртуальное окружение и убедитесь, что сервисы Docker работают.

1. **Шаг 1 — извлечение и нормализация тем**
   ```bash
   python scripts/pipeline-step_1-task_1.py   # извлечение тем из обзорных статей
   python scripts/pipeline-step_1-task_2.py   # векторизация и кластеризация тем
   python scripts/pipeline-step_1-task_3.py   # нормализация кластеров и канонические темы
   ```
2. **Шаг 2 — индексация корпуса для RAG**
   ```bash
   python scripts/pipeline-step_2-task_1.py
   ```
3. **Шаг 3 — генерация и дедупликация CQs**
   ```bash
   python scripts/pipeline-step3_cq_generation.py
   python scripts/pipeline-step3_llm_deduplication.py
   python scripts/pipeline-step3_supertopic_cq_generation.py
   ```
4. **Шаг 4 — финальные ответы по корпусу**
   ```bash
   python scripts/pipeline-step4_final_answers.py
   ```
5. **Шаг 6 — построение и очистка онтологии**
   ```bash
   python scripts/pipeline-step6_ontology_induction.py      # индукция TBox
   python scripts/pipeline-step6a_ontology_refinement.py    # агрессивное слияние классов
   python scripts/pipeline-step6b_ontology_canonization.py  # канонизация URI
   python scripts/pipeline-step6c_property_discovery.py     # поиск дополнительных свойств
   ```
6. **Шаг 7 — построение графа знаний (ABox)**
   ```bash
   python scripts/pipeline-step7_kg_pipeline.py
   ```

После выполнения этих шагов онтология (OWL) и граф знаний (KG) будут доступны в каталоге `outputs/<topic>/` (файлы `step6*/` и `step7/` соответственно).

## Результаты поиска научных статей

Результаты поиска литературы для подготовленного литературного обзора находятся в директории `corpus_screening_results`, где хранится итоговый анализ статей по трем тематическим направлениям.

Стадии скрининга:
- поиск по ключевым словам за период 2023-2025
- дополнительная фильтрация скаченного материала регулярными выражениями
- ручной скриниг


### CSV файлы с найденной информацией 

- autosurvey_includes_enriched.csv (49 статей)
- cq_includes_enriched.csv (102 статьи)
- ol_includes_enriched.csv (226 статей)

Колонки:
- `title` - название статьи
- `abstract` - аннотация
- `authors` - авторы
- `year` - год публикации
- `venue` - место публикации (журнал))
- `article_url` - URL статьи
- `pdf_url` - URL PDF
- `source_used` - источник данных (промежуточный файл сбора данных)
- `doi` - DOI
- `url` - URL
- `type` - тип
- `decision` - решение о включении (результат предыдущего автоматического скриненга)
- `reason` - обоснование решения (результат предыдущего автоматического скриненга)
