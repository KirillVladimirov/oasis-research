#!/bin/bash
# Скрипт для создания окружения для bigartm (Python 3.11)
# Устанавливает только библиотеки, необходимые для scripts/topic_modeling_bigartm

set -e

cd "$(dirname "$0")/.."

echo "Создание окружения для bigartm (Python 3.11)..."

# Удаляем старое окружение если есть
if [ -d ".venv-bigartm" ]; then
    echo "Удаление старого окружения .venv-bigartm..."
    rm -rf .venv-bigartm
fi

# Создаем новое окружение с Python 3.11
uv venv --python 3.11 .venv-bigartm

# Активируем окружение и устанавливаем только необходимые зависимости
source .venv-bigartm/bin/activate

echo "Установка зависимостей для BigARTM скриптов..."
uv pip install \
    loguru \
    tqdm \
    python-dotenv \
    openai \
    bigartm \
    "protobuf<4.0" \
    scikit-learn \
    scipy

deactivate

echo ""
echo "Окружение создано: .venv-bigartm"
echo "Активация: source .venv-bigartm/bin/activate"

