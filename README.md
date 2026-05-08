# 🚗 Car Price Prediction — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)
![MySQL](https://img.shields.io/badge/MySQL-Railway-4479A1?style=flat-square&logo=mysql)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A production-ready, end-to-end machine learning system that predicts second-hand car prices using **XGBoost**, served via a **FastAPI** REST API, containerised with **Docker**, deployed on **Render**, and backed by a **MySQL** database on **Railway**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development (Docker)](#local-development-docker)
  - [Local Development (Manual)](#local-development-manual)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Model Performance](#model-performance)
- [License](#license)

---

## Overview

This project demonstrates a complete machine learning lifecycle — from **raw data ingestion** through **preprocessing**, **feature engineering**, **model training**, and **hyperparameter-tuned prediction** — all exposed as a live REST API.

The system accepts a car's specifications (manufacturer, year, fuel type, engine, etc.) and returns a **predicted market price** in real-time. Every prediction is persisted to a cloud MySQL database for audit and analytics.

---

## Architecture

```
┌──────────────┐     POST /predict      ┌─────────────────────┐
│   API Client │ ──────────────────────▶│  FastAPI Application │
│  (Swagger /  │                        │     (Render)         │
│   Any HTTP)  │◀──────────────────────│                     │
└──────────────┘   predicted_price JSON └────────┬────────────┘
                                                 │
                              ┌──────────────────┼──────────────────┐
                              │                  │                  │
                    ┌─────────▼──────┐  ┌────────▼───────┐ ┌──────▼──────┐
                    │ Preprocessor   │  │  XGBoost Model │ │  MySQL DB   │
                    │  (.pkl)        │  │   (.pkl)       │ │  (Railway)  │
                    └────────────────┘  └────────────────┘ └─────────────┘
```

---

## ML Pipeline

The training pipeline is modular and orchestrated in sequential stages:

```
Raw CSV Data
    │
    ▼
1. Data Preprocessing   ← Clean levy/price/engine fields, remove nulls & duplicates
    │
    ▼
2. Data Ingestion       ← Train / Test split (80:20), save to artifacts/
    │
    ▼
3. Data Transformation  ← Encode categoricals (OHE), scale numerics, save preprocessor.pkl
    │
    ▼
4. Model Training       ← XGBRegressor (tuned with Optuna), save model.pkl
    │
    ▼
5. Prediction Pipeline  ← Load artifacts, clean input, transform, predict
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **ML Framework** | Scikit-learn, XGBoost |
| **Hyperparameter Tuning** | Optuna |
| **API Framework** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic v2 |
| **Database ORM** | SQLAlchemy 2.0 |
| **Database** | MySQL 9.4 (Railway) |
| **Containerisation** | Docker |
| **Deployment** | Render (Web Service) |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |

---

## Project Structure

```
car-price-prediction/
│
├── src/
│   ├── component/
│   │   ├── data_ingestion.py        # Load CSV, train/test split
│   │   ├── data_preprocessing.py   # Raw data cleaning pipeline
│   │   ├── data_transformation.py  # Feature encoding & scaling
│   │   └── model_trainer.py        # XGBoost training & evaluation
│   │
│   ├── pipline/
│   │   └── predict_pipline.py      # Inference pipeline + CarData schema
│   │
│   ├── exception.py                # Custom exception handler
│   ├── logger.py                   # Centralised logging
│   └── utils.py                    # save/load object helpers
│
├── notebook/
│   └── data/
│       └── car_price_prediction.csv  # Raw dataset
│
├── artifacts/                      # Generated by training (gitignored)
│   ├── model.pkl                   # Trained XGBoost model
│   ├── preprocessor.pkl            # Fitted ColumnTransformer
│   ├── train.csv
│   └── test.csv
│
├── app.py                          # FastAPI application entrypoint
├── database.py                     # SQLAlchemy engine & session
├── models.py                       # Car ORM model
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Local dev with MySQL
├── requirements.txt
└── setup.py
```

---

## API Reference

### Base URL
```
https://<your-render-service>.onrender.com
```

---

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Welcome to the Car Price Prediction API!"
}
```

---

### `POST /predict`
Submit car specifications and receive a predicted price.

**Request Body:**
```json
{
  "Levy": 500.0,
  "Manufacture": "Toyota",
  "Prod": "Corolla",
  "Year": 2018,
  "Category": "Sedan",
  "Leather": "Yes",
  "Fuel": "Petrol",
  "Gear": "Automatic",
  "Drive": "Front",
  "Engine": "1.6 Turbo",
  "Cylinders": 4,
  "Airbags": 6,
  "Doors": 4,
  "Wheel": "Left",
  "Color": "White"
}
```

**Response:**
```json
{
  "predicted_price": 14250.75
}
```

---

### `GET /cars/{car_id}`
Retrieve a previously predicted car record by ID.

**Response:**
```json
{
  "id": 1,
  "Manufacturer": "Toyota",
  "Prod_year": 2018,
  "Fuel_type": "Petrol",
  "predicted_price": 14250.75,
  ...
}
```

> 📖 Interactive API docs are available at `/docs` (Swagger UI) and `/redoc`.

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git

---

### Local Development (Docker)

The easiest way to run the full stack locally (FastAPI + MySQL) is with Docker Compose.

**1. Clone the repository**
```bash
git clone https://github.com/NazmulHudaNabil/end-to-end-ml-projects.git
cd end-to-end-ml-projects
```

**2. Create a `.env` file**
```bash
cp .env.example .env   # or create manually — see Environment Variables section
```

**3. Train the model** *(generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`)*
```bash
pip install -e .
python src/component/data_ingestion.py
```

**4. Start the application**
```bash
docker-compose up --build
```

The API will be live at **http://localhost:8000**  
Swagger UI: **http://localhost:8000/docs**

---

### Local Development (Manual)

**1. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**
```bash
pip install -e .
pip install -r requirements.txt
```

**3. Set environment variables**
```bash
export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/CarPricePrediction"
```

**4. Train the model**
```bash
python src/component/data_ingestion.py
```

**5. Run the API server**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Full MySQL connection string *(recommended)* | `mysql+pymysql://root:pass@host:3306/railway` |
| `DB_HOST` | MySQL host *(used if DATABASE_URL not set)* | `localhost` |
| `DB_PORT` | MySQL port | `3306` |
| `DB_NAME` | Database name | `CarPricePrediction` |
| `DB_USER` | Database username | `root` |
| `DB_PASSWORD` | Database password | `secret` |
| `PORT` | Port for the web server *(injected by Render)* | `10000` |

> ⚠️ **Never commit `.env` files or credentials to version control.**

---

## Deployment

This project is deployed using **Docker** on **Render** with a **MySQL** database hosted on **Railway**.

### Deploy to Render

1. Push the repository to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repository
4. Render auto-detects the `Dockerfile`
5. Set the following **Environment Variables** in the Render dashboard:
   - `DATABASE_URL` → *(your Railway MySQL public URL, with `mysql+pymysql://` prefix)*
6. Click **Deploy**

### Railway MySQL Setup

1. Go to [railway.app](https://railway.app) → **New Project → Add MySQL**
2. Navigate to **MySQL service → Variables tab**
3. Copy the `MYSQL_URL` value
4. Replace `mysql://` with `mysql+pymysql://`
5. Paste as `DATABASE_URL` in Render environment variables

---

## Model Performance

The **XGBoost Regressor** was trained with hyperparameters tuned using **Optuna**:

| Parameter | Value |
|---|---|
| `max_depth` | 8 |
| `learning_rate` | 0.03555 |
| `n_estimators` | 528 |
| `random_state` | 42 |

| Metric | Score |
|---|---|
| **Train R² Score** | > 0.90 |
| **Test R² Score** | > 0.85 |

> The model is rejected automatically during training if Test R² < 0.60, ensuring only quality models are deployed.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Nazmul Huda Nabil](https://github.com/NazmulHudaNabil)**

⭐ Star this repository if you found it helpful!

</div>
