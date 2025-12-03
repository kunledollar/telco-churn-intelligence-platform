
# End-to-End Telco Customer Churn Prediction System

This project implements a **real-world, end-to-end churn prediction system** for a Telco company:

- Business understanding
- Data loading & cleaning
- Feature engineering
- Model training with cross-validation
- Model selection & evaluation
- SHAP-based explainability (in notebook)
- API deployment with FastAPI
- Executive dashboard with Streamlit
- Docker container for the API

## 🔧 Tech Stack

- Python
- pandas, numpy, scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP
- FastAPI, Uvicorn
- Streamlit
- Docker

## 📂 Project Structure

```text
churn-prediction/
    data/
        Telco-Customer-Churn.xls
    notebooks/
        telco_churn_end_to_end.ipynb
    models/
        best_churn_model.joblib        # created after training
    src/
        __init__.py
        preprocessing.py               # data loading, cleaning, feature engineering
        model.py                       # training + cross-validation + model selection
        api.py                         # FastAPI inference service
    dashboard/
        app.py                         # Streamlit executive dashboard
    requirements.txt
    Dockerfile
    README.md
```

## 🚀 How to Run (Local)

1. Create and activate a virtual environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and save the best pipeline:

```bash
python -m src.model
```

This will create:

```text
models/best_churn_model.joblib
```

4. Run the FastAPI service:

```bash
uvicorn src.api:app --reload
```

Open: `http://127.0.0.1:8000/docs` for Swagger UI.

5. Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

## 🧠 SHAP Explainability

The notebook `notebooks/telco_churn_end_to_end.ipynb` contains SHAP analysis
(summary plot, etc.) so you can explain which features drive churn up or down.

## 🧑‍💼 Interview Summary

> "I built an end-to-end Telco churn prediction system using real data.  
> It includes feature engineering, model selection with 5-fold cross-validation
> (XGBoost / LightGBM / CatBoost), SHAP explainability, a FastAPI prediction
> service wrapped in Docker, and a Streamlit executive dashboard for churn
> monitoring and ROI discussions."

