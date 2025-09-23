from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "ml_pipeline",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0
}

dag = DAG(
    "dvc_pipeline",
    default_args=default_args,
    schedule="*/5 * * * *",  # every 5 minutes
    catchup=False
)

run_pipeline = BashOperator(
    task_id="run_dvc_pipeline",
    bash_command="cd /media/data/inno_courses/DL/as1/DL_as1 && dvc repro >> pipeline.log 2>&1",
    dag=dag
)
