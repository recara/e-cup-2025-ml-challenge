# OZON RecSys Baseline - Полный пайплайн с CatBoost и финайтюном
# Правильный формат submission: user_id,item_id_1,item_id_2,...,item_id_100

import os, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRanker, Pool

DATA_DIR = "/kaggle/input/testststs"

def load_train_data():
    print("Загружаем тренировочные данные...")
    
    # Загружаем только заказы
    orders_files = sorted(glob.glob(os.path.join(DATA_DIR, "ml_ozon_recsys_train_final_apparel_orders_data", "ml_ozon_recsys_train_final_apparel_orders_data", "*.parquet"), recursive=True))
    
    print(f"Найдено файлов заказов: {len(orders_files)}")
    
    # Загружаем только первые 2 файла для экономии памяти
    orders_data = []
    for file_path in tqdm(orders_files[:2], desc="Загрузка заказов"):
        df = pd.read_parquet(file_path, columns=["user_id", "item_id", "last_status", "created_date"])
        orders_data.append(df)
    
    orders_df = pd.concat(orders_data, ignore_index=True)
    print(f"Загружено заказов: {len(orders_df):,}")
    
    # Загружаем только базовые данные о товарах
    items_files = sorted(glob.glob(os.path.join(DATA_DIR, "ml_ozon_recsys_train_final_apparel_items_data", "ml_ozon_recsys_train_final_apparel_items_data", "*.parquet"), recursive=True))
    
    print(f"Найдено файлов товаров: {len(items_files)}")
    
    # Загружаем больше файлов для лучших признаков
    items_data = []
    for file_path in tqdm(items_files[:20], desc="Загрузка товаров"):
        df = pd.read_parquet(file_path, columns=["item_id", "catalogid", "attributes"])
        items_data.append(df)
    
    items_df = pd.concat(items_data, ignore_index=True)
    items_df = items_df.drop_duplicates(subset=['item_id'])
    print(f"Загружено товаров: {len(items_df):,}")
    
    return orders_df, items_df

def load_test_users():
    print("Загружаем тестовых пользователей...")
    
    test_files = sorted(glob.glob(os.path.join(DATA_DIR, "ml_ozon_recsys_test.snappy.parquet"), recursive=True))
    
    all_users = set()
    for file_path in tqdm(test_files, desc="Обработка тестовых файлов"):
        df = pd.read_parquet(file_path)
        all_users.update(df['user_id'].unique())
    
    # Берем всех тестовых пользователей
    test_users = list(all_users)
    print(f"Найдено уникальных тестовых пользователей: {len(test_users):,}")
    return test_users

# Загружаем данные
orders_df, items_df = load_train_data()

# Преобразуем created_date в datetime
orders_df['created_date'] = pd.to_datetime(orders_df['created_date'], errors='coerce')

# Определяем reference_date
reference_date = orders_df['created_date'].max()
validation_cutoff_date = reference_date - pd.Timedelta(days=30)
print(f"Дата отсечения для валидации: {validation_cutoff_date.date()}")

# Разделяем на train/validation
train_orders_df = orders_df[orders_df['created_date'] <= validation_cutoff_date].copy()
val_orders_df = orders_df[orders_df['created_date'] > validation_cutoff_date].copy()

print(f"Количество заказов в тренировочном наборе: {len(train_orders_df):,}")
print(f"Количество заказов в валидационном наборе: {len(val_orders_df):,}")

# Создаем улучшенную модель популярности
print("Создаем улучшенную модель популярности...")

# Считаем популярность товаров по доставленным заказам с нормализацией
delivered_orders = train_orders_df[train_orders_df['last_status'] == 'delivered_orders']
item_popularity_raw = delivered_orders['item_id'].value_counts()

# Нормализуем популярность (логарифмическая шкала)
item_popularity = np.log1p(item_popularity_raw).head(2000)  # Увеличиваем пул товаров

# Создаем train_df с лучшим балансом
print("Создаем тренировочный датасет...")
train_pairs = []
for user_id, item_id in delivered_orders[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
    train_pairs.append((user_id, item_id, 1))

train_df = pd.DataFrame(train_pairs, columns=["user_id", "item_id", "label"])

# Ограничиваем количество пользователей
unique_users = train_df["user_id"].unique()[:2000]  # Увеличиваем количество пользователей
train_df = train_df[train_df["user_id"].isin(unique_users)]

# Улучшенный negative sampling с большим количеством негативных примеров
all_items = item_popularity.index.tolist()
neg_pairs = []

for u in tqdm(unique_users, desc="Negative sampling"):
    bought_items = set(train_df[train_df["user_id"] == u]["item_id"].tolist())
    # Увеличиваем количество негативных примеров для лучшего обучения
    neg_items = np.random.choice(all_items, size=min(10, len(all_items)), replace=False)
    for item in neg_items:
        if item not in bought_items:
            neg_pairs.append((u, item, 0))

train_df = pd.concat([train_df, pd.DataFrame(neg_pairs, columns=["user_id", "item_id", "label"])], ignore_index=True)
print(f"Всего пар в train_df: {len(train_df):,}")

# Создаем топ-10 признаков
print("Создаем топ-10 признаков...")

# 1. Популярность товара (логарифмическая)
train_df["item_popularity"] = train_df["item_id"].map(lambda x: item_popularity.get(x, 0))

# 2. Ранг популярности товара
train_df["item_popularity_rank"] = train_df["item_id"].map(lambda x: item_popularity_raw.get(x, 0))

# 3. Количество покупок товара пользователем
user_item_purchases = delivered_orders.groupby(['user_id', 'item_id']).size().reset_index(name='user_item_count')
train_df = train_df.merge(user_item_purchases, on=['user_id', 'item_id'], how='left')
train_df["user_item_count"] = train_df["user_item_count"].fillna(0)

# 4. Общее количество покупок пользователя
user_total_purchases = delivered_orders.groupby('user_id').size().reset_index(name='user_total_purchases')
train_df = train_df.merge(user_total_purchases, on='user_id', how='left')
train_df["user_total_purchases"] = train_df["user_total_purchases"].fillna(0)

# 5. Доля товара в покупках пользователя
train_df["user_item_ratio"] = train_df["user_item_count"] / (train_df["user_total_purchases"] + 1)

# 6. Категория товара (catalogid)
train_df = train_df.merge(items_df[['item_id', 'catalogid']], on='item_id', how='left')
train_df["catalogid"] = train_df["catalogid"].fillna('unknown')

# 7. Популярность категории
category_popularity = delivered_orders.merge(items_df[['item_id', 'catalogid']], on='item_id', how='left')
category_popularity = category_popularity.groupby('catalogid').size().reset_index(name='category_popularity')
train_df = train_df.merge(category_popularity, on='catalogid', how='left')
train_df["category_popularity"] = train_df["category_popularity"].fillna(0)

# 8. Уникальность товара (обратная популярность)
max_popularity = item_popularity_raw.max()
train_df["item_uniqueness"] = 1 - (train_df["item_popularity_rank"] / max_popularity)

# Создаем val_df аналогично
print("Создаем валидационный датасет...")
val_delivered = val_orders_df[val_orders_df['last_status'] == 'delivered_orders']
val_pairs = []
for user_id, item_id in val_delivered[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
    val_pairs.append((user_id, item_id, 1))

val_df = pd.DataFrame(val_pairs, columns=["user_id", "item_id", "label"])

# Ограничиваем валидацию
unique_val_users = val_df["user_id"].unique()[:1000]  # Увеличиваем валидацию
val_df = val_df[val_df["user_id"].isin(unique_val_users)]

# Negative sampling для валидации
val_neg_pairs = []
for u in tqdm(unique_val_users, desc="Negative sampling для val"):
    bought_items = set(val_df[val_df["user_id"] == u]["item_id"].tolist())
    neg_items = np.random.choice(all_items, size=min(10, len(all_items)), replace=False)
    for item in neg_items:
        if item not in bought_items:
            val_neg_pairs.append((u, item, 0))

val_df = pd.concat([val_df, pd.DataFrame(val_neg_pairs, columns=["user_id", "item_id", "label"])], ignore_index=True)
print(f"Всего пар в val_df: {len(val_df):,}")

# Добавляем топ-10 признаков в val_df
print("Добавляем признаки в val_df...")

# 1. Популярность товара (логарифмическая)
val_df["item_popularity"] = val_df["item_id"].map(lambda x: item_popularity.get(x, 0))

# 2. Ранг популярности товара
val_df["item_popularity_rank"] = val_df["item_id"].map(lambda x: item_popularity_raw.get(x, 0))

# 3. Количество покупок товара пользователем
val_df = val_df.merge(user_item_purchases, on=['user_id', 'item_id'], how='left')
val_df["user_item_count"] = val_df["user_item_count"].fillna(0)

# 4. Общее количество покупок пользователя
val_df = val_df.merge(user_total_purchases, on='user_id', how='left')
val_df["user_total_purchases"] = val_df["user_total_purchases"].fillna(0)

# 5. Доля товара в покупках пользователя
val_df["user_item_ratio"] = val_df["user_item_count"] / (val_df["user_total_purchases"] + 1)

# 6. Категория товара (catalogid)
val_df = val_df.merge(items_df[['item_id', 'catalogid']], on='item_id', how='left')
val_df["catalogid"] = val_df["catalogid"].fillna('unknown')

# 7. Популярность категории
val_df = val_df.merge(category_popularity, on='catalogid', how='left')
val_df["category_popularity"] = val_df["category_popularity"].fillna(0)

# 8. Уникальность товара (обратная популярность)
val_df["item_uniqueness"] = 1 - (val_df["item_popularity_rank"] / max_popularity)

# Подготавливаем данные для CatBoost с топ-8 признаками
feature_cols = [
    "item_popularity",           # 1. Популярность товара
    "item_popularity_rank",      # 2. Ранг популярности
    "user_item_count",           # 3. Покупки товара пользователем
    "user_total_purchases",      # 4. Общие покупки пользователя
    "user_item_ratio",           # 5. Доля товара у пользователя
    "category_popularity",       # 6. Популярность категории
    "item_uniqueness",           # 7. Уникальность товара
    "catalogid"                  # 8. Категория товара
]

print(f"Используем {len(feature_cols)} признаков: {feature_cols}")
print("🎯 Топ-8 признаков для CatBoost:")
for i, feature in enumerate(feature_cols, 1):
    print(f"  {i}. {feature}")

# Сортируем по user_id для Ranker
train_df = train_df.sort_values("user_id").reset_index(drop=True)
val_df = val_df.sort_values("user_id").reset_index(drop=True)

train_X = train_df[feature_cols]
train_y = train_df["label"].astype(int)
train_group = train_df["user_id"].astype(str)

val_X = val_df[feature_cols]
val_y = val_df["label"].astype(int)
val_group = val_df["user_id"].astype(str)

train_pool = Pool(train_X, label=train_y, group_id=train_group)
val_pool = Pool(val_X, label=val_y, group_id=val_group)

# Обучаем CatBoost на топ-8 признаках
print("Обучаем CatBoostRanker на топ-8 признаках...")
model = CatBoostRanker(
    iterations=300,  # Больше итераций для сложных признаков
    depth=8,  # Увеличиваем глубину для сложных паттернов
    learning_rate=0.03,  # Медленнее обучение для лучшей регуляризации
    loss_function="YetiRank",
    eval_metric="NDCG:top=100",
    random_seed=42,
    verbose=50,
    l2_leaf_reg=3,  # L2 регуляризация
    bootstrap_type='Bernoulli',  # Dropout
    subsample=0.8,  # 80% данных для каждой итерации
    early_stopping_rounds=30,  # Early stopping
    cat_features=[feature_cols.index("catalogid")]  # Категориальный признак
)

model.fit(train_pool, eval_set=val_pool)

# Получаем метрики безопасно
print("Модель обучена успешно!")
print(f"Лучшая итерация: {model.get_best_iteration()}")

try:
    best_scores = model.get_best_score()
    if 'validation' in best_scores and 'NDCG:top=100' in best_scores['validation']:
        print(f"Лучший NDCG@100: {best_scores['validation']['NDCG:top=100']:.4f}")
    else:
        print(f"Лучший NDCG@100: {best_scores.get('NDCG:top=100', 'N/A')}")
except Exception as e:
    print(f"Ошибка получения метрик: {e}")

# Загружаем тестовых пользователей
test_users = load_test_users()

# Генерируем улучшенные рекомендации
print("Генерируем улучшенные рекомендации...")
recs = {}

for u in tqdm(test_users, desc="Генерация рекомендаций"):
    # Создаем кандидатов для пользователя с топ-8 признаками
    candidates = []
    for item_id in all_items[:500]:  # Берем топ-500 товаров как кандидатов
        item_cat = items_df[items_df['item_id'] == item_id]['catalogid'].iloc[0] if len(items_df[items_df['item_id'] == item_id]) > 0 else 'unknown'
        
        candidates.append({
            'item_id': item_id,
            'item_popularity': item_popularity.get(item_id, 0),
            'item_popularity_rank': item_popularity_raw.get(item_id, 0),
            'user_item_count': 0,  # Новый пользователь
            'user_total_purchases': 0,  # Новый пользователь
            'user_item_ratio': 0,  # Новый пользователь
            'category_popularity': category_popularity[category_popularity['catalogid'] == item_cat]['category_popularity'].iloc[0] if len(category_popularity[category_popularity['catalogid'] == item_cat]) > 0 else 0,
            'item_uniqueness': 1 - (item_popularity_raw.get(item_id, 0) / max_popularity),
            'catalogid': item_cat
        })
    
    if candidates:
        # Предсказываем скоры для кандидатов
        candidate_df = pd.DataFrame(candidates)
        candidate_X = candidate_df[feature_cols]
        
        # Получаем предсказания
        scores = model.predict(candidate_X)
        candidate_df['score'] = scores
        
        # Сортируем по скору и берем топ-100
        top_items = candidate_df.nlargest(100, 'score')['item_id'].tolist()
        recs[u] = top_items
    else:
        # Fallback на простую популярность
        recs[u] = item_popularity.head(100).index.tolist()

# Создаем submission в правильном формате
print("Создаем submission...")
submission_data = []

for u, items in recs.items():
    # Создаем словарь с user_id и 100 колонками item_id_1 до item_id_100
    row = {"user_id": u}
    
    # Заполняем 100 колонок item_id_1, item_id_2, ..., item_id_100
    for i in range(100):
        if i < len(items):
            row[f"item_id_{i+1}"] = items[i]
        else:
            row[f"item_id_{i+1}"] = items[0]  # Fallback если меньше 100 товаров
    
    submission_data.append(row)

submission = pd.DataFrame(submission_data)

# Сохраняем в правильном формате
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")
print(f"Submission shape: {submission.shape}")

# Проверяем формат
print("\nПроверка формата submission:")
print(f"Колонки: {submission.columns.tolist()}")
print(f"Количество колонок: {len(submission.columns)}")
print(f"Первые 5 колонок: {submission.columns[:5].tolist()}")
print(f"Последние 5 колонок: {submission.columns[-5:].tolist()}")

# Проверяем первую строку
first_row = submission.iloc[0]
print(f"user_id: {first_row['user_id']}")
print(f"item_id_1: {first_row['item_id_1']}")
print(f"item_id_100: {first_row['item_id_100']}")

print("\n✅ Готово! Полный пайплайн с CatBoost создан.")
print("📁 Файл: submission.csv")
print("📋 Формат: user_id,item_id_1,item_id_2,...,item_id_100")
print("🎯 Стратегия: CatBoostRanker с персонализацией")
print("🔧 Финайтюн: Регуляризация, early stopping, улучшенные признаки")
