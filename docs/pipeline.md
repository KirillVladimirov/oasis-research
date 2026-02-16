# Пайплайн OASIS

Конвейер генерации онтологии, компетентностных вопросов (CQ) и графа знаний. Шаги **1–8** и **10** образуют цепочку генерации; шаги **9** и **11** — замер метрик и не входят в саму генерацию (ответвляются в сторону).

---

## Схема пайплайна

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              Пайплайн генерации (сверху вниз)           │
                    └─────────────────────────────────────────────────────────┘

    Step 1  Download          Скачивание PDF датасетов
         │
         ▼
    Step 2  Extract text     Извлечение текста из PDF → topics/, rag/
         │
         ▼
    Step 3  Index Weaviate   Индексация чанков в Weaviate (эмбеддинги)
         │
         ▼
    Step 4  Prepare BigARTM  Подготовка к тематическому моделированию (VW, словарь)
         │
         ▼
    Step 5  Estimate K       Оценка числа тем K по датасетам
         │
         ▼
    Step 6  BigARTM          Обучение BigARTM и LLM-валидация тем
         │
         ▼
    Step 7  Generate CQs     Генерация компетентностных вопросов из тем
         │
         ▼
    Step 8  Score & ground   Скоринг CQ, grounding в Weaviate, построение онтологии
         │
         ├──────────────────────────►  Step 9  Eval      Замер метрик CQ и онтологии
         │
         ▼
    Step 10 Materialize KG   Материализация графа знаний (kg.ttl)
         │
         └──────────────────────────►  Step 11 KG Eval   Замер метрик качества KG
```

---

## Структура скриптов

| Шаг | Скрипт | Назначение |
|-----|--------|------------|
| **1** | `step1_download.py` | Скачивание PDF датасетов из конфига (GitHub Awesome и др.). Результат: `data/<dataset>/paper_pdfs/`. |
| **2** | `step2_extract_text.py` | Извлечение текста из PDF (парсер), формирование чанков. Выход: `data/<dataset>/topics/`, `rag/`. |
| **3** | `step3_index_weaviate.py` | Индексация датасетов в Weaviate: чанки, эмбеддинги. Нужен запущенный Weaviate. |
| **4** | `step4_prepare_bigartm.py` | Подготовка к BigARTM: VW-формат, словарь. Вход из `data/<dataset>`, выход в `artifacts/<dataset>/artm/`. |
| **5** | `step5_estimate_k.py` | Оценка числа тем K по датасетам. Результат: `artifacts/topic_counting/k_selection_all.json` (используется в шагах 6–7). |
| **6** | `step6_bigartm.py` | Обучение BigARTM (uni+bi), LLM-валидация тем. Выход: `artifacts/<dataset>/k<N>/` (темы, топики). |
| **7** | `step7_generate_cqs.py` | Генерация CQ из тем BigARTM (LLM по шаблонам S1). Выход: `artifacts/<dataset>/cqs.jsonl`, `cqs_scored.jsonl` и др. |
| **8** | `step8_score_ground_ontology.py` | Семантический скоринг CQ, grounding в Weaviate (evidence), построение OWL-онтологии. Выход: `cqs_grounded.jsonl`, `ontology.owl`. |
| **9** | `step9_eval.py` | **Замер метрик** (не часть генерации). Оценка CQ, онтологии и выравнивания CQ–онтология. Выход: `artifacts/<dataset>/eval/`. |
| **10** | `step10_kg_materialize.py` | Материализация графа знаний из `ontology.owl` и `cqs_grounded.jsonl`. Выход: `artifacts/<dataset>/kg/kg.ttl`, `kg_provenance.parquet`, `kg_stats.json`. |
| **11** | `step11_kg_eval.py` | **Замер метрик** (не часть генерации). Оценка качества KG. Выход: `artifacts/<dataset>/kg_eval/`, сводно `artifacts/kg_eval/kg_metrics_all.csv`. |

Конфигурация: `configs/pipeline.yaml`. Вспомогательный модуль: `scripts/pipeline/_config.py`.

---

## Вспомогательные скрипты (Weaviate, окружения)

| Скрипт | Назначение |
|--------|------------|
| **`scripts/check_weaviate_datasets.py`** | Проверка индексации датасетов в Weaviate: наличие классов `Document_<dataset>` и `Chunk_<dataset>`, число документов и чанков по каждому датасету. Опция `--weaviate-url`. Полезен после step 3 или после массовой индексации. |
| **`scripts/index_all_datasets.sh`** | Одним вызовом индексирует все перечисленные в скрипте датасеты в Weaviate (вызывает `step3_index_weaviate.py`). Переменные окружения: `WEAVIATE_URL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`. Запуск из корня проекта; приоритет Python: активный venv → `.venv-modern` → `.venv-bigartm` → `.venv`. |
| **`scripts/setup_env_modern.sh`** | Создаёт виртуальное окружение `.venv-modern` (Python 3.12) и ставит зависимости для «современного» стека (в т.ч. BERTopic, sentence-transformers, Weaviate). Опции: `--cuda` (PyTorch cu118), `--python X.Y`. Используется для шагов с эмбеддингами и Weaviate (step 3, 8 и др.). |
| **`scripts/setup_env_bigartm.sh`** | Создаёт окружение `.venv-bigartm` (Python 3.11) с зависимостями для BigARTM (bigartm, scikit-learn, scipy, loguru, openai и др.). Нужно для шагов 4–6 (подготовка корпуса, оценка K, обучение BigARTM). |

---

## Краткое описание шагов генерации

- **Step 1 — Download.** Для каждого датасета из конфига берётся конфиг топика (URL репозитория GitHub Awesome, опционально anchor секции в README). Скачивается raw README, из него вырезается нужная секция по anchor, из markdown извлекаются ссылки (\[...](url), голые https). Для каждой ссылки определяется источник (arXiv, прямая PDF, OpenReview) и URL PDF. Файлы скачиваются в `data/<dataset>/paper_pdfs/`; уже существующие пропускаются. Возвращаются счётчики: found, downloaded, skipped, errors.

- **Step 2 — Extract text.** Для каждого датасета обходится каталог `paper_pdfs/`. Каждый PDF обрабатывается парсером (структурированное извлечение: блоки текста, таблицы, отсечение headers/footers и секции references). Формируются два текстовых представления статьи: для топиков и для RAG. Результаты пишутся в `data/<dataset>/topics/` (файлы `*_topics.txt`) и `data/<dataset>/rag/` (файлы `*_rag.txt`). Скрипт возвращает число успешно обработанных и число ошибок.

- **Step 3 — Index Weaviate.** По каждому датасету читаются тексты из `data/<dataset>/` (topics или rag). Тексты разбиваются на чанки с заданными `chunk_size` и `chunk_overlap` из конфига. Для чанков считаются эмбеддинги (модель из конфига, по умолчанию bge-m3). Объекты с полями chunk_id, paper_id, text и вектором загружаются в Weaviate по указанному URL. Индекс используется на шаге 8 для семантического поиска evidence при grounding CQ.

- **Step 4 — Prepare BigARTM.** Для каждого датасета из `data/<dataset>/` читаются текстовые файлы (например, из topics). Текст токенизируется (нижний регистр, удаление URL/email, стоп-слова, короткие токены; опционально отсекаются библиографические термины). Строятся униграммы и при необходимости биграммы/триграммы; токены санитизируются для формата Vowpal Wabbit (запрещённые символы заменяются). Генерируется `vw.txt` (одна строка на документ: doc_id |text ... |bigrams ...) и в `preprocess/` — tokenized.jsonl, doc_map.json, vocab_stats.json. Затем собирается словарь BigARTM из VW-корпуса и сохраняется в `artifacts/<dataset>/artm/` вместе с батчами.

- **Step 5 — Estimate K.** По каждому датасету загружается корпус из директории с `.txt` файлами (токенизация без стоп-слов). Строятся статистики совместной встречаемости (document frequency, pair document frequency). Для набора значений K от k_min до k_max с шагом k_step выполняется несколько bootstrap-запусков BigARTM на подвыборке корпуса. Для каждого K считаются coherence (NPMI по топ-токенам тем), diversity (уникальность слов в топах), stability (сходство тем между запусками по Jaccard). K выбирается по взвешенной комбинации метрик. Результаты записываются в `artifacts/topic_counting/k_selection_all.json`.

- **Step 6 — BigARTM.** Для каждого датасета с заданным K (из step 5) выполняется обучение модели ARTM: загрузка словаря и батчей из `artifacts/<dataset>/artm/`, создание тематических компонент с настройками разреживания (theta, phi, decorrelator), обучение offline на заданное число проходов. Сохраняются матрицы и топ-токены по темам. Затем запускается LLM-валидация тем: для каждой темы по топ-токенам формируется запрос к LLM на оценку осмысленности/связности; результаты и темы пишутся в `artifacts/<dataset>/k<K>/` (topics_llm.jsonl и др.).

- **Step 7 — Generate CQs.** По каждому датасету вызывается скрипт генерации CQ из тем BigARTM (`01_generate_cqs_from_topics_bigartm.py`). На вход подаётся файл тем `k<K>/<dataset>--k<K>--topics_llm.jsonl`. По темам и квотам по ролям (class, relation, process, constraint, comparison) формируются промпты к LLM; при необходимости запрашиваются релевантные фрагменты из Weaviate для контекста. LLM генерирует компетентностные вопросы; выход постобрабатывается и сохраняется в `artifacts/<dataset>/cqs.jsonl`. Опционально выполняется скоринг и получается `cqs_scored.jsonl`.

- **Step 8 — Score, ground, ontology.** Для каждого датасета по цепочке запускаются три подшага (каждый — отдельный subprocess). (1) **Scoring:** `02_cq_semantic_scoring.py` — по `cqs.jsonl` вычисляется семантический скоринг CQ, результат в `cqs_scored.jsonl`. (2) **Grounding:** `03_cq_grounding_weaviate.py` — для каждого CQ выполняется запрос к Weaviate по тексту вопроса, возвращаются top-k чанков (evidence) с метаданными и score; результат в `cqs_grounded.jsonl`. (3) **Ontology:** `04_build_ontology_from_cqs.py` — по `cqs_grounded.jsonl` извлекаются кандидаты классов и свойств из формулировок и evidence, выполняется слияние меток (эмбеддинги/LLM), строится OWL (классы, объектные/дата-свойства, иерархия) и экспортируется в `ontology.owl`.

- **Step 10 — Materialize KG.** Читаются `ontology.owl` и `cqs_grounded.jsonl` по датасету. OWL загружается в граф (rdflib), подгружается TBox-манифест для справки. Для каждой записи CQ создаётся узел CQ с текстом; для каждого evidence_chunk — узел Chunk (chunk_id, doc_id, опционально text_span), узел Paper (doc_id, при наличии — title, year, venue из references.csv), связь Chunk → inPaper → Paper. Для каждой пары (CQ, evidence) создаётся реифицированный узел Support (BNode) с полями fromCQ, toChunk, score и `prov:wasDerivedFrom` на Chunk; CQ связывается с Chunk через supportedBy. Граф сериализуется в Turtle в `artifacts/<dataset>/kg/kg.ttl`; дополнительно пишутся kg_provenance.parquet (или csv) и kg_stats.json.

Шаги **9** и **11** — только замер метрик по уже полученным артефактам. Метрики описаны в [evaluation_metrics.md](evaluation_metrics.md).
