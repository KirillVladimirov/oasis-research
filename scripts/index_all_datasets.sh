#!/bin/bash
# Скрипт для индексации всех датасетов в Weaviate

# Не прерываем выполнение при ошибках - обрабатываем их вручную
set +e

# Определяем Python из виртуального окружения, если оно активно
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python3"
elif [ -f ".venv-modern/bin/python3" ]; then
    PYTHON=".venv-modern/bin/python3"
elif [ -f ".venv-bigartm/bin/python3" ]; then
    PYTHON=".venv-bigartm/bin/python3"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

WEAVIATE_URL="${WEAVIATE_URL:-http://localhost:8081}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-models/bge-m3}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-50}"

DATASETS=(
    "ml_systems"
    "deep_learning_neural_networks"
    "diffusion_models"
    "generative_models_in_pathology"
    "graph_nn_research"
    "graph_nn_systems"
    "healthcare_foundation_models"
    "ml_systems_distributed_training"
    "ml_systems_inference_system"
    "ml_systems_mixture_of_experts_moe"
    "model_quantization"
    "multimodal_ai"
    "speech_recognition_synthesis"
    "deep_active_learning"
    "causal_inference"
    "automl"
)

echo "Starting indexing of ${#DATASETS[@]} datasets..."
echo "Weaviate URL: $WEAVIATE_URL"
echo "Embedding model: $EMBEDDING_MODEL"
echo ""

# Запуск из корня проекта; логика в oasis.pipelines.index_weaviate (step3)
if "$PYTHON" scripts/pipeline/step3_index_weaviate.py \
    --weaviate-url "$WEAVIATE_URL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --chunk-size "$CHUNK_SIZE" \
    --chunk-overlap "$CHUNK_OVERLAP" \
    --datasets "${DATASETS[@]}"; then
    echo ""
    echo "All datasets indexed successfully!"
else
    echo ""
    echo "Indexing failed for one or more datasets."
    exit 1
fi
