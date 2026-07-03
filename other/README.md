# Ozon ML Challenge 2025 - Исправленный Бейзлайн (24_08_25)

## Описание

Этот репозиторий содержит исправленный бейзлайн для задачи Ozon ML Challenge 2025. Этот файл (`ml_ozon_recsys_fixed_24_08_25.py`) является прямым преобразованием ноутбука `ml-ozon-recsys-24 08 25.ipynb` с исправленными путями к файлам данных, адаптированными под структуру датасета Kaggle.

## Структура данных

Датасет Kaggle (`/kaggle/input/testststs`) имеет следующую структуру:

- **Items**: `ml_ozon_recsys_train_final_apparel_items_data/ml_ozon_recsys_train_final_apparel_items_data/*.parquet`
- **Orders**: `ml_ozon_recsys_train_final_apparel_orders_data/ml_ozon_recsys_train_final_apparel_orders_data/*.parquet`
- **Tracker**: `ml_ozon_recsys_train_final_apparel_tracker_data/ml_ozon_recsys_train_final_apparel_tracker_data/*.parquet`
- **Test**: `ml_ozon_recsys_test.snappy.parquet`
- **Categories**: `ml_ozon_recsys_train_final_categories_tree/ml_ozon_recsys_train_final_categories_tree/*.parquet`

## Исправления в путях

В исходном ноутбуке пути были жестко закодированы или использовали неполные `glob` паттерны. В этом файле они были исправлены для корректной загрузки данных из указанной структуры Kaggle:

### Было (пример для orders):
```python
orders_files = glob.glob('/kaggle/input/testststs/ml_ozon_recsys_train_final_apparel_orders_data/ml_ozon_recsys_train_final_apparel_orders_data/*.parquet')
```

### Стало (пример для orders):
```python
orders_files = sorted(glob.glob(os.path.join(DATA_DIR, "ml_ozon_recsys_train_final_apparel_orders_data", "ml_ozon_recsys_train_final_apparel_orders_data", "*.parquet"), recursive=True))
```

И аналогичные изменения были применены ко всем остальным файлам данных (`items`, `tracker`, `test`, `categories`).

## Файлы в репозитории

- `ml_ozon_recsys_fixed_24_08_25.py` - Исправленный Python скрипт бейзлайна.
- `README.md` - Этот файл с инструкциями.

## Запуск

### Локально с Poetry:

Для воспроизведения проекта используйте Poetry:

1.  **Клонируйте репозиторий** (если еще не сделали это):
    ```bash
    git clone <ваш_репозиторий>
    cd <ваш_репозиторий>
    ```
2.  **Установите зависимости**:
    ```bash
    poetry install
    ```
3.  **Активируйте виртуальное окружение Poetry**:
    ```bash
    poetry shell
    ```
4.  **Разместите данные Kaggle**: Убедитесь, что данные датасета Kaggle (`https://www.kaggle.com/datasets/thunderpede/testststs`) расположены по пути `/kaggle/input/testststs` (или измените `DATA_DIR` в `ml_ozon_recsys_fixed_24_08_25.py` на ваш локальный путь к данным).
5.  **Запустите скрипт**:
    ```bash
    python ml_ozon_recsys_fixed_24_08_25.py
    ```

### В Kaggle Notebook:

1.  Убедитесь, что ваш ноутбук имеет доступ к датасету Kaggle: `https://www.kaggle.com/datasets/thunderpede/testststs`.
2.  Откройте `ml_ozon_recsys_fixed_24_08_25.py`.
3.  Скопируйте все содержимое файла.
4.  Вставьте его в новую ячейку вашего Kaggle Notebook.
5.  Запустите ячейку.

## Особенности бейзлайна

Этот пайплайн представляет собой значительно улучшенное базовое решение для задачи Ozon ML Challenge 2025. Он включает в себя следующие ключевые особенности и улучшения:

-   **Расширенный Feature Engineering**:
    -   **Признаки товаров**: Извлечены и агрегированы категориальные атрибуты товаров (`attribute_name`, `attribute_value`).
    -   **Признаки категорий**: Добавлен новый признак `category_depth` на основе иерархии категорий.
    -   **Признаки взаимодействий**: Агрегированные признаки на уровне пользователя (`user_interaction_count`, `user_unique_items_interacted`, `user_recency`) и на уровне товара (`item_interaction_count`, `item_unique_users_interacted`) из данных трекера.
    -   **CLIP-эмбеддинги**: Интегрированы 512-мерные CLIP-эмбеддинги изображений товаров в качестве числовых признаков.
-   **Улучшенная стратегия генерации кандидатов**: Используется персонализированный подход, основанный на недавней истории пользователя и категориях этих товаров, в дополнение к глобально популярным товарам.
-   **Модель ранжирования CatBoostRanker**:
    -   Используется `CatBoostRanker` с функцией потерь `YetiRankPair` и метрикой оценки `NDCG:top=100`.
    -   **Тонкая настройка гиперпараметров**: Реализован ручной перебор нескольких комбинаций гиперпараметров (`iterations`, `depth`, `learning_rate`, `l2_leaf_reg`) с использованием `early_stopping_rounds` для поиска лучшей модели.
    -   **Анализ важности признаков**: Выводится отсортированный список и визуализация важности признаков, что позволяет понять вклад каждого признака в предсказания модели.
    -   **Явная оценка метрик**: После обучения лучшей модели выводится ее метрика `NDCG@100` на валидационном наборе.
-   **Временная валидация**: Данные `orders` разделяются на обучающий и валидационный наборы по дате (`validation_cutoff_date`), что обеспечивает более реалистичную оценку производительности модели.
-   **Оптимизированная обработка больших данных**: Использование библиотеки `polars` для эффективной загрузки `orders`, `items` и `categories` файлов, что сокращает потребление памяти и время загрузки.
-   **Формат submission**: Генерирует файл `baseline_submission.csv` в требуемом формате.
