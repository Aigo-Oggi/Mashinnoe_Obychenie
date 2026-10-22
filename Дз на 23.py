# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:16:41 2026

@author: Артём_Б
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ

# Читаем файл (поменяйте имя на ваше, если нужно)
df = pd.read_csv('credit_risk_dataset.csv')

# Заполняем пропуски медианами (базовый инструмент Pandas)
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())


# 2. ПОСТРОЕНИЕ ГРАФИКОВ БАНКА (Через чистый Matplotlib)
plt.figure(figsize=(10, 5))

# ГРАФИК: Кредитная нагрузка у хороших заемщиков (0) и дефолтных (1)
# Разделяем заемщиков на две группы
good_clients = df[df['loan_status'] == 0]['loan_percent_income']
bad_clients = df[df['loan_status'] == 1]['loan_percent_income']

# Строим гистограмму распределения долговой нагрузки
plt.hist(good_clients, bins=30, alpha=0.5, label='Успешный возврат (0)', color='blue', density=True)
plt.hist(bad_clients, bins=30, alpha=0.5, label='Дефолт / Убыток банка (1)', color='red', density=True)

plt.title('Как долговая нагрузка (доля от дохода) губит заемщиков банка')
plt.xlabel('Доля платежа по кредиту от общего дохода заемщика')
plt.ylabel('Плотность клиентов')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 3. АЛГОРИТМ КЛАССИФИКАЦИИ ВРУЧНУЮ (Бизнес-правило банка)

# Вместо сложных нейросетей банк часто использует скоринговые правила.
# Проверим классическое правило риск-менеджмента: 
# "Если у человека БЫЛ дефолт в истории ИЛИ его кредитная нагрузка > 25% — банк считает его неплательщиком (1)".

# Перемешиваем и делим данные на тест вручную (70/30)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df_shuffled) * 0.7)
train_df = df_shuffled.iloc[:split_idx]
test_df = df_shuffled.iloc[split_idx:].copy()

# Применяем наше скоринговое правило на тестовой выборке
test_df['predicted_status'] = np.where(
    (test_df['cb_person_default_on_file'] == 'Y') | (test_df['loan_percent_income'] > 0.25), 
    1, 0
)

# Считаем точность (Accuracy) нашего правила вручную
correct_predictions = (test_df['predicted_status'] == test_df['loan_status']).sum()
accuracy = correct_predictions / len(test_df)

print("\n=== РЕЗУЛЬТАТЫ СКОРИНГА БАНКА ВРУЧНУЮ ===")
print(f"Общая точность бизнес-правила (Accuracy): {accuracy * 100:.2f}%")
