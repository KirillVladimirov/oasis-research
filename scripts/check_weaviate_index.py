#!/usr/bin/env python3
"""Проверка индексов и количества данных в Weaviate."""

import sys
from pathlib import Path

import weaviate
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

WEAVIATE_URL = "http://localhost:8081"


def main() -> int:
    """Проверяет схему и количество объектов в Weaviate."""
    logger.info("=" * 80)
    logger.info("ПРОВЕРКА ИНДЕКСОВ WEAVIATE")
    logger.info("=" * 80)

    try:
        client = weaviate.Client(WEAVIATE_URL, startup_period=10)
        logger.info(f"Подключено к Weaviate: {WEAVIATE_URL}")
    except Exception as e:
        logger.error(f"Ошибка подключения к Weaviate: {e}")
        logger.error("Убедитесь, что Weaviate запущен: docker compose up -d weaviate")
        logger.info("\nАльтернативные команды curl:")
        logger.info(f"  curl {WEAVIATE_URL}/v1/schema")
        logger.info(f'  curl -X POST {WEAVIATE_URL}/v1/graphql -H "Content-Type: application/json" -d \'{{"query": "{{ Aggregate {{ Chunk {{ meta {{ count }} }} }} }}"}}\'')
        return 1

    # Получаем схему
    try:
        schema = client.schema.get()
        classes = schema.get("classes", [])
        
        if not classes:
            logger.warning("В Weaviate нет классов (индексов)")
            return 0
        
        logger.info(f"\nНайдено классов: {len(classes)}")
        logger.info("-" * 80)
        
        expected_classes = ["Paper", "Chunk"]
        
        # Для каждого класса показываем схему и количество объектов
        for class_info in classes:
            class_name = class_info.get("class", "Unknown")
            
            if class_name not in expected_classes:
                logger.warning(f"\nНайден устаревший класс: {class_name}")
                logger.warning("  Класс не используется, рекомендуется удалить (индексация с delete_existing=True)")
                continue
            
            logger.info(f"\n Класс: {class_name}")
            
            properties = class_info.get("properties", [])
            if properties:
                logger.info(f"  Свойства ({len(properties)}):")
                for prop in properties[:7]:
                    prop_name = prop.get("name", "?")
                    prop_type = prop.get("dataType", ["?"])[0]
                    logger.info(f"    - {prop_name}: {prop_type}")
                if len(properties) > 7:
                    logger.info(f"    ... и еще {len(properties) - 7} свойств")
            
            if class_name == "Chunk":
                has_parent_ref = any(
                    prop.get("name") == "parent" and "Paper" in prop.get("dataType", [])
                    for prop in properties
                )
                if has_parent_ref:
                    logger.info("   Связь: parent -> Paper (parent-child)")
                else:
                    logger.warning("   Связь parent -> Paper не найдена")
            
            try:
                result = client.query.aggregate(class_name).with_meta_count().do()
                count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
                logger.info(f"   Количество объектов: {count:,}")
            except Exception as e:
                logger.warning(f"   Не удалось подсчитать объекты: {e}")
        
        # Проверка эмбеддингов в классе Chunk
        logger.info("\n" + "=" * 80)
        logger.info("ПРОВЕРКА ЭМБЕДДИНГОВ")
        logger.info("=" * 80)
        
        try:
            # Получаем несколько объектов Chunk с векторами
            result = client.query.get("Chunk", ["chunk_id", "text", "paper_id", "page_start", "page_end"]).with_additional(["vector"]).with_limit(5).do()
            chunks = result.get("data", {}).get("Get", {}).get("Chunk", [])
            
            if chunks:
                vectors_found = 0
                vectors_missing = 0
                
                for chunk in chunks:
                    additional = chunk.get("_additional", {})
                    vector = additional.get("vector", [])
                    
                    if vector:
                        vectors_found += 1
                        if vectors_found == 1:  # Показываем детали только для первого
                            logger.success(" Эмбеддинги найдены")
                            logger.info(f"  - Размерность вектора: {len(vector)}")
                            logger.info(f"  - Первые 5 значений: {[round(v, 4) for v in vector[:5]]}")
                            logger.info(f"  - Пример чанка ID: {chunk.get('chunk_id', 'N/A')}")
                            logger.info(f"  - paper_id: {chunk.get('paper_id', 'N/A')}")
                            logger.info(f"  - Страницы: {chunk.get('page_start', 'N/A')}-{chunk.get('page_end', 'N/A')}")
                            logger.info(f"  - Текст (первые 100 символов): {chunk.get('text', '')[:100]}...")
                    else:
                        vectors_missing += 1
                
                if vectors_found > 0:
                    logger.info(f"\nСтатистика по проверенным чанкам:")
                    logger.info(f"  С векторами: {vectors_found}")
                    if vectors_missing > 0:
                        logger.warning(f"  Без векторов: {vectors_missing}")
                else:
                    logger.warning("Векторы не найдены в объектах Chunk")
            else:
                logger.warning("Объекты Chunk не найдены в Weaviate")
                logger.info("Возможно, индексация не завершена или данные удалены")
        except Exception as e:
            logger.warning(f"Ошибка проверки эмбеддингов: {e}")
        
        # Проверка связей Chunk -> Paper
        logger.info("\n" + "=" * 80)
        logger.info("ПРОВЕРКА СВЯЗЕЙ (Chunk -> Paper)")
        logger.info("=" * 80)
        
        try:
            # Получаем один Chunk с ссылкой на Paper
            result = client.query.get("Chunk", ["chunk_id", "paper_id"]).with_additional(["id"]).with_limit(1).do()
            chunks = result.get("data", {}).get("Get", {}).get("Chunk", [])
            
            if chunks:
                chunk = chunks[0]
                chunk_id = chunk.get("chunk_id", "N/A")
                paper_id = chunk.get("paper_id", "N/A")
                
                # Пытаемся получить Paper через ссылку parent
                try:
                    # Проверяем наличие Paper с таким paper_id
                    paper_result = client.query.get("Paper", ["paper_id", "title"]).with_where({
                        "path": ["paper_id"],
                        "operator": "Equal",
                        "valueText": paper_id
                    }).with_limit(1).do()
                    
                    papers = paper_result.get("data", {}).get("Get", {}).get("Paper", [])
                    if papers:
                        logger.success(" Связь Chunk -> Paper работает")
                        logger.info(f"  - Chunk: {chunk_id}")
                        logger.info(f"  - Paper: {paper_id} ({papers[0].get('title', 'N/A')[:50]}...)")
                    else:
                        logger.warning(f"Paper с paper_id='{paper_id}' не найден")
                except Exception as e:
                    logger.debug(f"Ошибка проверки связи: {e}")
            else:
                logger.warning("Не удалось проверить связи: нет объектов Chunk")
        except Exception as e:
            logger.debug(f"Ошибка проверки связей: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.success("Проверка завершена")
        return 0
        
    except Exception as e:
        logger.exception(f"Ошибка при проверке схемы: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
