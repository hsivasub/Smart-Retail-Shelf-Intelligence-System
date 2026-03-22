# Smart Retail Shelf Intelligence System

An end-to-end AI system that analyzes retail shelf images to detect empty shelf slots, misplaced products, and calculate overall shelf health scores.

## Architecture overview
- **Computer Vision Model**: YOLOv8 (Object Detection for empty slots & products)
- **Anomaly Detection**: Isolation Forest (Misplaced items detection based on visual embeddings/coordinates)
- **API Engine**: FastAPI
- **Experiment Tracking**: MLflow
- **Pipelines**: Apache Airflow
- **Monitoring**: Evidently (Data Drift)
- **Dashboard**: Streamlit
