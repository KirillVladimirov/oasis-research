#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEAVIATE_DATA_DIR="$PROJECT_ROOT/weaviate_data"

echo "Исправление прав доступа к директории Weaviate..."
echo "Директория: $WEAVIATE_DATA_DIR"

if docker ps | grep -q weaviate; then
    echo "Остановка контейнера weaviate..."
    docker stop weaviate
    docker rm weaviate
fi

if [ -d "$WEAVIATE_DATA_DIR" ]; then
    echo "Удаление старой директории (требуются права sudo)..."
    sudo rm -rf "$WEAVIATE_DATA_DIR"
fi

echo "Создание новой директории..."
mkdir -p "$WEAVIATE_DATA_DIR"

echo "Права установлены. Теперь можно запустить:"
echo "  docker compose up -d weaviate"
echo "или"
echo "  docker-compose up -d weaviate"
