from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define default arguments for the DAG operations
default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    'smart_shelf_training_pipeline',
    default_args=default_args,
    description='End-to-end Airflow DAG to ingest data, preprocess, and train YOLO & Anomaly models.',
    schedule_interval=timedelta(days=1), # Retrain daily
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['retail_shelf', 'mlops', 'training'],
)

# Task 1: Data Ingestion
# Simulates pulling the latest images from the source storage bucket
t1_ingestion = BashOperator(
    task_id='data_ingestion',
    bash_command='python src/pipelines/data_ingestion.py',
    dag=dag,
)

# Task 2: Preprocessing
# Resizes, norms, and crops using OpenCV
t2_preprocessing = BashOperator(
    task_id='image_preprocessing',
    bash_command='python src/pipelines/preprocessing.py',
    dag=dag,
)

# Task 3: Train Object Detection
# YOLOv8 fine-tuning logged to MLflow
t3_train_yolo = BashOperator(
    task_id='train_yolo_detection',
    bash_command='python src/detection/train.py',
    dag=dag,
)

# Task 4: Train Anomaly Model
# Updates the Isolation Forest with new coordinate patterns
t4_train_anomaly = BashOperator(
    task_id='train_isolation_forest',
    # Since model is a class, we simulate its training trigger
    bash_command='python -c "import numpy; from src.anomaly.model import ShelfAnomalyDetector; detector=ShelfAnomalyDetector(); detector.train(numpy.random.rand(10,6))"',
    dag=dag,
)

# Set execution flow constraints
t1_ingestion >> t2_preprocessing >> t3_train_yolo >> t4_train_anomaly
