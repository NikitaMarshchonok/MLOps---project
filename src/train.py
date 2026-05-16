import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Говорим MLFlow куда отправлять данные
# localhost:5001 - это наш MLflow сервер в докер!
mlflow.set_tracking_uri('http://localhost:5001')

# название эксперимента
mlflow.set_experiment('churn_prediction')


# Создаём искусственные данные о клиентах
# В реальном проекте это были бы данные из базы
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'age':np.random.randint(18, 70, n_customers),
    'months_subscribed': np.random.randint(1, 60, n_customers),
    'monthly_charge': np.random.uniform(10, 100, n_customers),
    'support_calls': np.random.randint(0,10, n_customers),
    'churn': np.random.randint(0, 2, n_customers),
})


# Делим данные на признаки (X) и целевую переменную (y)
# X — то что знаем о клиенте
# y — то что хотим предсказать (ушёл или нет)
X = data.drop('churn', axis=1)
y = data['churn']

# Делим на тренировочные и тестовые данные
# 80% — учимся, 20% — проверяем
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size =0.2, random_state=42
)

# mlflow.start_run() — открываем "страницу в журнале"
# Всё что внутри with — записывается в MLflow
with mlflow.start_run():
    # Параметры модели  - записываем что использовали

    n_estimators = 100
    max_depth = 5
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # Cоздаем и обучаем модель
    model = RandomForestClassifier(
        n_estimators = n_estimators, 
        max_depth =max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)


# Проверяем на тестовых данных
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

#Записываем метрики в MLflow
mlflow.log_metric('accuracy', accuracy)
mlflow.log_metric('f1_score', f1)

# сохраняем саму модель 
mlflow.sklearn.log_model(model, 'random_forest_model')

print(f"Accuracy:{accuracy:.3f}")
print(f"F1 Score:{f1:.3f}")
print("Эксперимент записан в MLFlow!")
