# AutoScraping Arxiv Paper and Multi-label Categories Classification

<div align="center">
<img src="[https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)" />
<img src="[https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white](https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white)" />
<img src="[https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)" />
<img src="[https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)" />
<img src="[https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Python-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite)" />
</div>

<div align="center">

**🔥 A robust Airflow pipeline for scraping Arxiv papers, processing data, and performing multi-label classification using MongoDB and MinIO.**

[🚀 Quick Start](https://www.google.com/search?q=%23quick-start) | [✨ Features](https://www.google.com/search?q=%23features) | [🤖 Machine Learning](https://www.google.com/search?q=%23machine-learning) | [📋 Management](https://www.google.com/search?q=%23management-commands) | [🛠️ Troubleshooting](https://www.google.com/search?q=%23troubleshooting)

</div>

## 📋 Overview

This project is a comprehensive **Apache Airflow** implementation designed to automate the extraction and classification of research papers. It orchestrates a pipeline that scrapes data from Arxiv, cleans and normalizes it, stores metadata in **MongoDB**, saves artifacts in **MinIO**, and trains a multi-label classification model using **Scikit-Learn**.

## 🚀 Quick Start

Follow these steps to set up and run the application locally.

### 1. 📥 Prerequisites

Ensure you have the following installed:

* **Docker Desktop** (running)
* **Git Bash** or **WSL2** (for Windows users to run scripts)

### 2. ⚙️ Configuration

Create a `.env` file in the project root:

```ini
# .env
# Airflow Admin User
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_ADMIN_EMAIL=admin@example.com

# Airflow configuration (Only for information)
AIRFLOW_UID=50000
AIRFLOW_IMAGE_NAME=apache/airflow:3.1.1rc1-python3.10

# Database
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

```

### 3. 📂 Environment Setup

Create the necessary directories for logs, plugins, and temporary data:

```bash
mkdir logs
mkdir plugins
mkdir -p tmp/arxiv_data

```

### 4. 🐳 Build and Initialize

Build the Docker image and initialize the database:

```bash
docker compose build
docker compose up airflow-init

```

### 5. 🔄 Launch Services

Start the Airflow services, Database, and MinIO in detached mode:

```bash
docker compose up -d

```

### 6. 🌐 Access Services

Once up and running, access the services at the following URLs:

| Service | URL | Credentials |
| --- | --- | --- |
| **Airflow UI** | `http://localhost:8080` | `admin` / `admin` |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin123` |
| **MongoDB** | `localhost:27017` | `admin` / `admin123` (DB: `arxiv_db`) |

## ✨ Features

### 🔄 Data Pipeline (ETL)

* **🤖 AutoScraping**: Automated fetching of papers from Arxiv.
* **🧹 Data Cleaning**:
* **Deduplication**: Removes papers with identical IDs.
* **Normalization**: Trims whitespace and removes invalid special characters.
* **Validation**: Verifies PDF URLs and ensures date format (`YYYY-MM-DD`).
* **Quality Checks**: Flags critical missing fields (ID/Title) and adds `data_quality` tracking.



### 📊 Data Management

* **MinIO Integration**: Object storage for large artifacts.
* **MongoDB Storage**: Structured storage for paper metadata.
* *Connection String*: `mongodb://admin:admin123@localhost:27017/`


* **Local Outputs**: Cleaned CSVs available in `tmp/arxiv_data/`.

## 🤖 Machine Learning

The project includes a dedicated DAG (`arxiv_category_trainer`) for predicting paper categories.

### 🧠 Model Details

* **Task**: Multi-label Classification (predicting multiple categories per paper).
* **Algorithm**: `OneVsRestClassifier` wrapping `LogisticRegression`.
* **Features**: TF-IDF Vectorization of Titles and Abstracts.
* **Artifacts**: Saved locally in `tmp/ml_training/models/`.
* `model.joblib`: Trained classifier.
* `vectorizer.joblib`: TF-IDF vectorizer.
* `label_encoder.joblib`: Multi-label binarizer.



### 🏃‍♂️ Running Inference

To test the model locally:

```bash
python inference.py

```

## 📋 Management Commands

### 🚦 Controlling Airflow

**Stop services (preserve data):**

```bash
docker compose down
# OR
bash docker-stop.sh

```

**Restart services:**

```bash
docker compose restart

```

**Delete completely (reset database):**

```bash
docker compose down -v
docker compose up airflow-init
docker compose up -d

```

### 📝 Logging

**View all logs:**

```bash
docker compose logs -f

```

**View specific service logs:**

```bash
docker compose logs -f airflow-scheduler

```

## 🔌 DAG Usage

1. Navigate to `http://localhost:8080` and log in.
2. Locate **`arxiv_paper_scraper`** for data collection or **`arxiv_category_trainer`** for ML.
3. Toggle the DAG to **ON**.
4. Click the **Play Button** (Trigger DAG) to run manually.

## 🛠️ Troubleshooting

### 🔍 Common Issues

* **🚫 Invalid Auth Token / Signature Verification Failed**:
JWT secrets may be out of sync.
* **Fix**:
```bash
docker-compose down
docker-compose up -d

```


* *Verify*: Run the python check command (see original docs) to ensure `api_auth` secrets match between scheduler and webserver containers.


* **🐳 Cannot connect to Docker daemon**:
Ensure Docker Desktop is running.
* **🚪 Port 8080 already in use**:
Update `docker-compose.yaml` to use a different port:
```yaml
ports:
  - "8081:8080"

```


* **🔒 Permission Denied (Linux/Mac)**:
```bash
chmod +x docker-start.sh docker-stop.sh

```


* **👻 DAG Not Appearing**:
Force a reserialization:
```bash
docker exec airflowsimple-airflow-scheduler-1 airflow dags reserialize

```



## 🔒 Security

> **⚠️ production warning**: Never use default credentials in production.

Update your `.env` file before deploying:

```ini
AIRFLOW_ADMIN_USERNAME=your_secure_user
AIRFLOW_ADMIN_PASSWORD=your_strong_password
AIRFLOW__API_AUTH__JWT_SECRET=your_generated_secret_key

```