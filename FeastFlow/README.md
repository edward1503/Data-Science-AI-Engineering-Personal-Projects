# FeastFlow - End-to-End ML Pipeline with Feast Feature Store

🚀 **Project Overview**  
FeastFlow is a comprehensive demonstration of a modern MLOps pipeline built around [Feast](https://feast.dev/), the open-source feature store. This project showcases a production-ready machine learning system designed to address real-world challenges such as training-serving skew, feature redundancy, and operational complexity.

**Primary Focus**: Demonstrate the power of Feast as the central feature management platform in an end-to-end ML workflow.

🎯 **Key Features**

- 📊 **Complete ETL Pipeline**: From raw data ingestion to engineered features.
- 🏪 **Feast Integration**: Offline and online feature stores powered by Redis.
- 🤖 **ML Training & Inference**: Point-in-time correct training and real-time predictions.
- 🎯 **Streamlit Dashboard**: Interactive UI to explore the entire pipeline.
- 🔗 **Version Control**: Git and DVC for complete reproducibility.
- 🐳 **Containerization**: Docker support for seamless deployment.

🏗️ **Architecture**  
![Architecture Diagram](https://raw.githubusercontent.com/dangnha/FeastFlow/master/static/pipeline.png) _(Placeholder for architecture diagram)_

📁 **Project Structure**

```
feastflow-demo/
├── data/                    # Datasets
│   ├── raw/                # Raw dataset from Kaggle
│   └── processed/          # Transformed and cleaned data
├── feature_repo/            # Feast feature store
│   ├── feature_store.yaml  # Feast configuration
│   ├── features.py         # Feature definitions
│   └── data/               # Feast registry
├── scripts/                 # Pipeline scripts
│   ├── download_data.py    # Data extraction from Kaggle
│   ├── transform_data.py   # Feature engineering
│   ├── setup_feast.py      # Feast initialization
│   └── train_model.py      # Model training
├── api/                     # Inference service
│   └── app.py              # FastAPI service
├── model/                   # Trained models and metadata
├── streamlit_app.py         # Interactive dashboard
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Multi-service setup
└── run.sh                  # One-click demo script
```

🛠️ **Technologies Used**

- **Feature Store**: Feast 0.31.0
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Database**: Redis (online store), Parquet files (offline store)
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Version Control**: Git, DVC
- **Containerization**: Docker, Docker Compose
- **Visualization**: Plotly

📋 **Prerequisites**

- Python 3.8+
- Docker and Docker Compose
- Git
- Kaggle API credentials for downloading data

🚀 **Quick Start**

```bash
# Clone the repository
git clone https://github.com/dangnha/FeastFlow.git>
cd feastflow-demo

# Install dependencies
pip install -r requirements.txt

# Run the complete demo
bash run.sh
```

**Note**:

- Ensure Docker is running (e.g., open Docker Desktop on Windows).
- Set up Kaggle API credentials in your environment (see [Kaggle API documentation](https://www.kaggle.com/docs/api)).

📊 **Dataset**  
This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, which includes:

- 7,043 customers with 21 features
- A mix of numerical and categorical data
- A clear business problem: churn prediction
- Temporal elements ideal for Feast demonstrations

🎮 **Using the Demo**

**Streamlit Dashboard**  
Access the interactive dashboard at: [http://localhost:8501](http://localhost:8501)

Features:

- 🔍 Select any customer from the dataset.
- 🤖 Get real-time churn predictions.
- 🏪 View features served from the Feast online store.
- 📊 Explore pipeline visualizations and metrics.
- 🎯 Understand feature importance.

**API Endpoints**

- **Health Check**: `GET http://localhost:8000/health`
- **Prediction**:
  ```bash
  POST http://localhost:8000/predict
  {
      "customer_id": "1234-ABCD"
  }
  ```
- **Feature Retrieval**: `GET http://localhost:8000/features/1234-ABCD`
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

🏪 **Feast Implementation**  
**Feature Views**:

- **Customer Demographics**: Static customer information.
- **Customer Financials**: Monthly charges, tenure, and totals.
- **Customer Contract**: Service and contract details.

**Key Feast Concepts Demonstrated**:

- **Entities**: Customer as the primary entity.
- **Feature Views**: Logical grouping of features.
- **Offline Store**: Historical features for training.
- **Online Store**: Low-latency feature serving.
- **Point-in-Time Correctness**: Prevents data leakage.
- **Feature Materialization**: Keeps the online store updated.

🔧 **Development**  
**Adding New Features**:

1. Update `feature_repo/features.py`.
2. Apply changes: `feast apply`.
3. Materialize to the online store: `feast materialize`.

**Model Retraining**:

```bash
python scripts/train_model.py
```

Built with ❤️ for the AIO2025 community.
