# 🚨 Credit Card Fraud Detection System (MLOps + Production Ready)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

An **end-to-end Machine Learning system** for detecting fraudulent credit card transactions in real-time.

Built with a **production-first mindset**, this project integrates:

* ✅ Machine Learning (XGBoost)
* ✅ FastAPI for real-time predictions
* ✅ Docker for deployment
* ✅ Logging & Monitoring (MLOps)
* ✅ Threshold-based risk scoring

---

## 🎯 Problem Statement

Fraud detection is a **high-risk, high-impact problem**:

* Fraud cases are **extremely rare (~0.17%)**
* Dataset is **highly imbalanced**
* Missing fraud (**False Negative**) is very costly

---

## 🧠 Solution Approach

Instead of optimizing for accuracy, this system focuses on:

* 🔥 **High Recall (Fraud Detection Priority)**
* 📉 **PR-AUC over Accuracy**
* ⚖️ **Class Imbalance Handling (SMOTE)**
* 🎯 **Custom Threshold Optimization**

---

## 📊 Dataset

* 📦 Source: Kaggle Credit Card Fraud Dataset
* 🔢 Transactions: ~284,000
* 🚨 Fraud Cases: ~492

### Key Features:

* `V1–V28`: PCA-transformed features
* `Amount`: Only interpretable feature
* No missing values

---

## 🔍 EDA Insights

* Severe class imbalance (<1% fraud)
* Amount distribution is highly skewed
* PCA removes multicollinearity

---

## ⚙️ Model Selection

### 🚀 XGBoost

Why XGBoost?

* Handles **imbalanced data effectively**
* Captures **non-linear patterns**
* Strong performance on tabular data
* Built-in regularization

---

## ⚖️ Handling Imbalance

### ✅ SMOTE (Synthetic Minority Oversampling)

* Generates synthetic fraud samples
* Avoids data loss (better than undersampling)

---

## 🎯 Threshold Optimization (Key Innovation)

### ❌ Default (0.5)

* Misses fraud cases

### ✅ Custom Threshold (~0.03)

* Maximizes **Recall (~91%)**

💡 **Business Insight:**

> Missing fraud is more expensive than false positives.

---

## 📈 Model Performance

| Metric    | Value                      |
| --------- | -------------------------- |
| ROC-AUC   | ~0.98                      |
| PR-AUC    | ~0.87                      |
| Recall    | ~0.91                      |
| Precision | Lower (expected trade-off) |

---

## 🏗️ System Architecture

```text
User → FastAPI → Preprocessing → Model → Threshold → Response
```

---

## 🚀 Features

* 🔥 Real-time Fraud Detection API
* 📊 Risk Classification (SAFE / MEDIUM / HIGH)
* 🧾 Prediction Logging (`logs/predictions.csv`)
* 🧩 Modular & Scalable Codebase
* 🐳 Dockerized Deployment

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── app/
│   ├── main.py
│   ├── routes/predict.py
│   ├── schemas/
│   ├── utils/
│   ├── model/
│
├── monitoring/
├── logs/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 🔹 1. Clone Repo

```bash
git clone <repo-link>
cd fraud-detection-system
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Run API

```bash
uvicorn app.main:app --reload
```

### 🔹 4. Open API Docs

```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## 📡 API Example

### 🔹 Input

```json
{
  "Time": 0,
  "V1": -1.35,
  "V2": -0.07,
  "V3": 2.53,
  "V4": 1.37,
  "Amount": 149.62
}
```

### 🔹 Output

```json
{
  "fraud": false,
  "probability": 0.03,
  "threshold_used": 0.034,
  "risk_level": "SAFE",
  "message": "Safe Transaction ✅"
}
```

---

## 📊 Logging & Monitoring

* Predictions stored in:

```
logs/predictions.csv
```

### Why this matters:

* 📈 Monitor model performance
* 🔍 Debug incorrect predictions
* 🔄 Use data for retraining
* ⚠️ Detect data drift

---

## ⚠️ Important Notes

* Dataset not included (size constraints)
* Model files included for inference
* Logs excluded via `.gitignore`

---

## 🧠 Key Learnings

* Handling **imbalanced datasets**
* Importance of **PR-AUC**
* Threshold tuning for business impact
* Building **production-ready ML systems**
* MLOps practices (logging, monitoring)

---

## 🚀 Future Improvements

* ☁️ Cloud Deployment (AWS / GCP)
* 📊 Monitoring Dashboard
* 🔄 CI/CD Pipeline
* ⚡ Real-time Streaming (Kafka)
* 🧠 Drift Detection (Evidently)

---

## 🎯 Resume Highlight

> Built a production-ready fraud detection system using XGBoost, SMOTE, and FastAPI with recall-optimized threshold tuning for highly imbalanced data.

---

## 👨‍💻 Author

**Lohith Reddy Gayam**
