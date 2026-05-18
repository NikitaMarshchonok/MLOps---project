from datetime import timedelta
import pendulum
import subprocess
from airflow.decorators import dag, task

default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="model_retraining",
    schedule="0 3 * * 1",        # каждый понедельник в 3 ночи
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training"],
)
def model_retraining():

    @task()
    def retrain_model():
        result = subprocess.run(
            ["python", "/opt/airflow/src/train.py"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise Exception(f"Обучение упало: {result.stderr}")
        print("Модель успешно переобучена!")

    retrain_model()

dag_instance = model_retraining()