#!/usr/bin/env python
# coding: utf-8

# # Анализ и подготовка данных
# 
# 

# ## Описание исходного набора данных

# In[1]:


# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('retail_store_inventory.csv')


# ## Структура и типы данных

# In[2]:


print("="*32)
print("ОБЩАЯ ИНФОРМАЦИЯ О НАБОРЕ ДАННЫХ")
print("="*32)

print(f"Размерность данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 5 строк данных:")
print(df.head())
print("\nИнформация о типах данных и пропусках:")
df.info()


# ### Описание признаков

# | Признак            | Тип    | Описание                       |
# | ------------------ | ------ | ------------------------------ |
# | Date               | object | дата наблюдения                |
# | Store ID           | object | идентификатор магазина         |
# | Product ID         | object | идентификатор товара           |
# | Category           | object | категория товара               |
# | Region             | object | регион                         |
# | Inventory Level    | int    | уровень запасов                |
# | Units Sold         | int    | количество проданных единиц    |
# | Units Ordered      | int    | количество заказанных единиц   |
# | Demand Forecast    | float  | предварительный прогноз спроса |
# | Price              | float  | цена товара                    |
# | Discount           | int    | величина скидки                |
# | Weather Condition  | object | погодные условия               |
# | Holiday/Promotion  | int    | признак праздника или акции    |
# | Competitor Pricing | float  | цена конкурента                |
# | Seasonality        | object | сезонность                     |
# 

# ## Определение целевой переменной и типов признаков

# In[3]:


# Определение целевой переменной
TARGET_COLUMN = 'Units Sold'
print(f"1. Целевая переменная: '{TARGET_COLUMN}'")
print("   - Количество проданных единиц товара")
print("   - Непосредственно отражает потребительский спрос")
print("   - Метрика для оценки качества прогноза")

# Идентификационные признаки (не используются как предикторы)
ID_COLUMNS = ['Store ID', 'Product ID']
print(f"\n2. Идентификационные признаки: {ID_COLUMNS}")
print("   - Используются только для группировки данных")
print("   - Не интерпретируются как числовые признаки")

# Временной признак
DATE_COLUMN = 'Date'
print(f"\n3. Временной признак: '{DATE_COLUMN}'")
print("   - Определяет временную структуру данных")
print("   - Используется для формирования временных рядов")

# Категориальные признаки
categorical_columns = [
    'Store ID', 'Product ID', 'Category', 'Region',
    'Weather Condition', 'Seasonality'
]
print(f"\n4. Категориальные признаки: {categorical_columns}")
print(f"   - Количество: {len(categorical_columns)}")

# Числовые признаки
numerical_columns = [
    'Units Sold', 'Units Ordered', 'Price', 'Discount',
    'Holiday/Promotion', 'Competitor Pricing'
]
print(f"\n5. Числовые признаки: {numerical_columns}")
print(f"   - Количество: {len(numerical_columns)}")


# ## Преобразование типов данных

# In[4]:


# Преобразование временного признака
df['Date'] = pd.to_datetime(df['Date'])

# Преобразование категориальных признаков
for col in categorical_columns:
    if col in df.columns:
        unique_count = df[col].nunique()
        df[col] = df[col].astype('category')

print("\nПроверка преобразованных типов данных:")
print(df.dtypes.to_string())


# ## Анализ и обработка пропущенных значений

# In[5]:


missing_values = df.isna().sum()
print(f"Количество пропущенных значений: \n{missing_values}")


# Анализ показал, что пропущенные значения в датасете отсутствуют, что исключает необходимость применения методов восстановления данных.

# ## Анализ дубликатов

# In[6]:


# Поиск полных дубликатов
full_duplicates = df.duplicated().sum()
print(f"Количество полных дубликатов строк: {full_duplicates}")

# Поиск дубликатов по ключевым полям (дата-магазин-товар)
key_columns = ['Date', 'Store ID', 'Product ID']
date_duplicates = df.duplicated(subset=key_columns).sum()
print(f"\nКоличество дубликатов по ключевым полям {key_columns}: {date_duplicates}")


# ## Анализ целевой переменной

# In[7]:


# Базовая статистика
target_stats = df['Units Sold'].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("Статистика целевой переменной Units Sold")
print(target_stats.to_string())

# Дополнительные метрики
print(f"\nДополнительные метрики:")
print(f"Коэффициент вариации: {df['Units Sold'].std() / df['Units Sold'].mean() * 100:.2f}%")
print(f"Асимметрия: {df['Units Sold'].skew():.4f}")
print(f"Эксцесс: {df['Units Sold'].kurtosis():.4f}")

# Визуализация распределения
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['Units Sold'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Units Sold')
axes[0].set_ylabel('Частота')
axes[0].set_title('Гистограмма распределения Units Sold')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df['Units Sold'], vert=False)
axes[1].set_xlabel('Units Sold')
axes[1].set_title('Boxplot Units Sold')

from scipy import stats
stats.probplot(df['Units Sold'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot (сравнение с нормальным распределением)')

plt.tight_layout()
plt.show()

# Анализ нулевых и экстремальных значений
zero_sales = (df['Units Sold'] == 0).sum()
print(f"Дней с нулевыми продажами: {zero_sales} ({zero_sales/len(df)*100:.2f}%)")
print(f"Максимальное значение: {df['Units Sold'].max()}")


# ## Проверка логической согласованности данных

# In[8]:


# Проверка корректности цен
price_issues = {}
price_issues['negative_prices'] = (df['Price'] <= 0).sum()
price_issues['negative_competitor_prices'] = (df['Competitor Pricing'] <= 0).sum()

print("Проверка ценовых признаков:")
for issue, count in price_issues.items():
    print(f"   {issue}: {count} строк")

# Сравнение проданных и заказанных единиц
print("\nСравнение Units Sold и Units Ordered:")
sold_exceeds_ordered = (df['Units Sold'] > df['Units Ordered']).sum()
print(f"Продажи превышают заказы в {sold_exceeds_ordered} случаях "
      f"({sold_exceeds_ordered/len(df)*100:.2f}%)")

# Анализ причин превышения продаж над заказами
if sold_exceeds_ordered > 0:
    excess_cases = df[df['Units Sold'] > df['Units Ordered']]
    print(f"\n   Анализ случаев превышения продаж над заказами:")
    print(f"   Среднее превышение: {(excess_cases['Units Sold'] - excess_cases['Units Ordered']).mean():.2f}")
    print(f"   Максимальное превышение: {(excess_cases['Units Sold'] - excess_cases['Units Ordered']).max()}")


# Превышение на 51.4% - это большое значение, которое, видимо, вызывано проблемой в данных.

# ## Детальный анализ проблемы с превышением продаж на 51.4%

# In[9]:


# Создаем колонку для анализа
df['excess_sales'] = df['Units Sold'] - df['Units Ordered']

# Статистика превышения
print("\n1. СТАТИСТИКА ПРЕВЫШЕНИЯ:")
excess_stats = df[df['excess_sales'] > 0]['excess_sales'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(excess_stats.to_string())

# Распределение: сколько случаев с небольшим превышением, сколько с большим?
print("\n2. РАСПРЕДЕЛЕНИЕ ПО ВЕЛИЧИНЕ ПРЕВЫШЕНИЯ:")
bins = [0, 10, 50, 100, 200, 500]
labels = ['0-10', '11-50', '51-100', '101-200', '>200']
df['excess_category'] = pd.cut(df[df['excess_sales'] > 0]['excess_sales'], bins=bins, labels=labels)

excess_distribution = df[df['excess_sales'] > 0]['excess_category'].value_counts().sort_index()
for category, count in excess_distribution.items():
    percentage = count / sold_exceeds_ordered * 100
    print(f"  Превышение {category}: {count} случаев ({percentage:.1f}%)")

# Есть ли проблемные магазины?
print("\n3. АНАЛИЗ ПО МАГАЗИНАМ (TOP-10 по количеству аномалий):")
store_issues = df[df['excess_sales'] > 0].groupby('Store ID').size().sort_values(ascending=False).head(10)
for store, count in store_issues.items():
    store_total = df[df['Store ID'] == store].shape[0]
    percentage = count / store_total * 100
    print(f"  Магазин {store}: {count} аномалий из {store_total} записей ({percentage:.1f}%)")

# По товарам
print("\n4. АНАЛИЗ ПО ТОВАРАМ (TOP-10):")
product_issues = df[df['excess_sales'] > 0].groupby('Product ID').size().sort_values(ascending=False).head(10)
for product, count in product_issues.items():
    product_total = df[df['Product ID'] == product].shape[0]
    percentage = count / product_total * 100
    print(f"  Товар {product}: {count} аномалий из {product_total} записей ({percentage:.1f}%)")

# По времени (дни недели, месяцы)
print("\n5. АНАЛИЗ ПО ВРЕМЕНИ:")
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month

# По дням недели
print("  По дням недели:")
for day in range(7):
    day_data = df[df['day_of_week'] == day]
    anomalies = (day_data['Units Sold'] > day_data['Units Ordered']).sum()
    total = len(day_data)
    percentage = anomalies / total * 100 if total > 0 else 0
    day_name = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'][day]
    print(f"    {day_name}: {anomalies}/{total} ({percentage:.1f}%)")

# Корреляция со скидками и ценами
print("\n6. КОРРЕЛЯЦИЯ С ДРУГИМИ ПРИЗНАКАМИ:")
print("  Связь со скидками:")
discount_anomalies = df[(df['excess_sales'] > 0) & (df['Discount'] > 0)].shape[0]
print(f"    Аномалии со скидкой: {discount_anomalies} ({discount_anomalies/sold_exceeds_ordered*100:.1f}%)")

print("  Связь с праздниками:")
holiday_anomalies = df[(df['excess_sales'] > 0) & (df['Holiday/Promotion'] == 1)].shape[0]
print(f"    Аномалии в праздники: {holiday_anomalies} ({holiday_anomalies/sold_exceeds_ordered*100:.1f}%)")


# ### Визуализация проблемы

# In[10]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Гистограмма превышения
axes[0, 0].hist(df[df['excess_sales'] > 0]['excess_sales'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Превышение продаж над заказами')
axes[0, 0].set_ylabel('Количество случаев')
axes[0, 0].set_title('Распределение величины превышения')
axes[0, 0].axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Порог 100')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Boxplot превышения по магазинам (топ-5)
top_stores = df[df['excess_sales'] > 0].groupby('Store ID').size().nlargest(5).index
store_data = df[df['Store ID'].isin(top_stores) & (df['excess_sales'] > 0)]
sns.boxplot(data=store_data, x='Store ID', y='excess_sales', ax=axes[0, 1])
axes[0, 1].set_title('Распределение превышения по магазинам (топ-5)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].set_ylabel('Превышение')

# Тепловая карта по дням недели и месяцам
pivot_table = df.pivot_table(
    values='excess_sales',
    index='month',
    columns='day_of_week',
    aggfunc=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
)
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 2])
axes[0, 2].set_title('% аномалий по месяцам и дням недели')
axes[0, 2].set_xlabel('День недели (0=Пн)')
axes[0, 2].set_ylabel('Месяц')

# Динамика аномалий во времени
daily_anomalies = df.set_index('Date')['excess_sales'].apply(lambda x: 1 if x > 0 else 0)
daily_anomalies_resampled = daily_anomalies.resample('W').mean() * 100  # Процент аномалий в неделю
axes[1, 0].plot(daily_anomalies_resampled.index, daily_anomalies_resampled.values)
axes[1, 0].set_xlabel('Дата')
axes[1, 0].set_ylabel('% аномальных дней')
axes[1, 0].set_title('Динамика аномалий во времени (по неделям)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')

# Связь со скидками
discount_groups = pd.cut(df['Discount'], bins=[-1, 0, 5, 10, 15, 20])
discount_analysis = df.groupby(discount_groups).apply(
    lambda x: (x['Units Sold'] > x['Units Ordered']).sum() / len(x) * 100
)
discount_analysis.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_xlabel('Уровень скидки')
axes[1, 1].set_ylabel('% аномалий')
axes[1, 1].set_title('Влияние скидок на аномалии')
axes[1, 1].tick_params(axis='x', rotation=45)

# Scatter plot: продажи vs заказы
axes[1, 2].scatter(df['Units Ordered'], df['Units Sold'], alpha=0.3, s=10)
axes[1, 2].plot([0, df['Units Ordered'].max()], [0, df['Units Ordered'].max()],
               'r--', label='y=x (продажи=заказы)')
axes[1, 2].set_xlabel('Units Ordered')
axes[1, 2].set_ylabel('Units Sold')
axes[1, 2].set_title('Продажи vs Заказы (все точки)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Анализ аномалии: продажи > заказов (51.84% случаев)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ### Методологическое решение проблемы

# In[11]:


print("\nАНАЛИЗ ВАРИАНТОВ ОБРАБОТКИ:")

# Вариант 1: Полное исключение признака
print("\n1. ПОЛНОЕ ИСКЛЮЧЕНИЕ Units Ordered:")
print("   Плюсы:")
print("   - Устраняет проблему полностью")
print("   - Упрощает модель")
print("   Минусы:")
print("   - Теряем потенциально полезную информацию")
print("   - Units Ordered может содержать прогнозную информацию")

# Вариант 2: Коррекция данных
print("\n2. КОРРЕКЦИЯ ДАННЫХ (Units_Ordered_Corrected = max(продажи, заказы)):")
print("   Плюсы:")
print("   - Логически корректные данные")
print("   - Сохраняем информацию о заказах")
print("   - Простое и понятное решение")
print("   Минусы:")
print("   - Изменяет исходные данные")
print("   - Может искажать статистику заказов")

# Вариант 3: Две версии данных
print("\n3. ДВЕ ВЕРСИИ ДАННЫХ (с коррекцией и без):")
print("   Плюсы:")
print("   - Научная строгость (сравнение подходов)")
print("   - Позволяет оценить влияние коррекции")
print("   Минусы:")
print("   - Удваивает объем работы")
print("   - Усложняет анализ")


# Я буду реализовывать третий вариант, так как он является более строгим и не изменяет исходные данные.

# ### Создание двух версий данных

# In[12]:


from scipy import stats

# Создаем копии данных для двух подходов
df_version_a = df.copy()  # Версия A: с коррекцией (основная)
df_version_b = df.copy()  # Версия B: без коррекции (контрольная)

print("\nВЕРСИЯ A (С КОРРЕКЦИЕЙ):")
print("Units_Ordered_Corrected = np.maximum(Units Ordered, Units Sold)")
df_version_a['Units_Ordered_Corrected'] = np.maximum(df_version_a['Units Ordered'], df_version_a['Units Sold'])
df_version_a['Was_Corrected'] = (df_version_a['Units_Ordered_Corrected'] != df_version_a['Units Ordered']).astype(int)

print("\nВЕРСИЯ B (БЕЗ КОРРЕКЦИИ):")
print("Сохраняем исходные данные без изменений")
df_version_b['Units_Ordered_Original'] = df_version_b['Units Ordered'].copy()
df_version_b['Has_Anomaly'] = (df_version_b['Units Sold'] > df_version_b['Units Ordered']).astype(int)

# Статистическое сравнение двух версий
print("\n" + "="*36)
print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ ДВУХ ВЕРСИЙ")
print("="*36)

comparison_stats = pd.DataFrame(index=['Версия A (с коррекцией)', 'Версия B (без коррекции)'])

# Базовые статистики
comparison_stats['Записи с аномалиями'] = [
    df_version_a['Was_Corrected'].sum(),
    df_version_b['Has_Anomaly'].sum()
]

comparison_stats['% аномалий'] = [
    df_version_a['Was_Corrected'].mean() * 100,
    df_version_b['Has_Anomaly'].mean() * 100
]

comparison_stats['Среднее Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].mean(),
    df_version_b['Units Ordered'].mean()
]

comparison_stats['Стд. отклонение Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].std(),
    df_version_b['Units Ordered'].std()
]

comparison_stats['Медиана Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].median(),
    df_version_b['Units Ordered'].median()
]

comparison_stats['Min Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].min(),
    df_version_b['Units Ordered'].min()
]

comparison_stats['Max Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].max(),
    df_version_b['Units Ordered'].max()
]

print(comparison_stats.round(2).to_string())


# Рассматриваются две версии данных:
# 
# Версия A (с коррекцией):
#     Units_Ordered_Corrected = max(Units Ordered, Units Sold). Данный подход устраняет логическое противоречие, однако использует целевую переменную (Units Sold) при формировании признаков, что приводит к утечке информации и делает модель неприменимой для реального прогнозирования.
# 
# Версия B (без коррекции):
#     Исходные данные сохраняются без изменений. Аномалия интерпретируется как продажи из складских остатков и отражает фактический потребительский спрос.
# 
# В дальнейшем для моделирования используется только Версия А.

# ### Создание копий данных для двух подходов

# In[13]:


df_version_a = df.copy()
df_version_b = df.copy()


# In[14]:


# Рабочий датасет для анализа и моделирования
df = df_version_b.copy()


# In[15]:


df_version_a['Units_Ordered_Corrected'] = np.maximum(
    df_version_a['Units Ordered'],
    df_version_a['Units Sold']
)

# заменяем Units Ordered на скорректированную
df_version_a['Units Ordered'] = df_version_a['Units_Ordered_Corrected']
df_version_a = df_version_a.drop(columns=['Units_Ordered_Corrected'])


# ### Исключение признаков с утечкой информации

# In[16]:


leakage_cols = ['Demand Forecast', 'Inventory Level']

df_version_a = df_version_a.drop(columns=leakage_cols, errors='ignore')
df_version_b = df_version_b.drop(columns=leakage_cols, errors='ignore')


# Demand Forecast - это заранее рассчитанный прогноз спроса.
# 
# Inventory Level отражает ограничения предложения, а не потребительский спрос.

# ## Итоговая проверка данных

# In[17]:


print("VERSION A:")
print(df_version_a[['Units Sold','Units Ordered']].describe())

print("\nVERSION B (without corrections):")
print(df_version_b[['Units Sold','Units Ordered']].describe())


# In[18]:


df_version_a.to_csv("final_version_A.csv", index=False)
df_version_b.to_csv("final_version_B.csv", index=False)

print("  final_version_A.csv  — version with anomaly correction")
print("  final_version_B.csv  — working version without correction")

