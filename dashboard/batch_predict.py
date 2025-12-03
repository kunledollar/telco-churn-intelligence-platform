import pandas as pd
import joblib

def load_model(model_path="/app/dashboard/models/best_churn_model.joblib"):
    """
    Loads the trained churn model.
    """
    return joblib.load(model_path)

def preprocess_batch(df):
    """
    Recreates full feature engineering used in training,
    ensuring all required model features exist.
    """

    df = df.copy()

    # ---- Clean basic data ----
    df = df.fillna(0)

    # Ensure numeric types
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # ---- 1. monthly_to_total_ratio ----
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)

    # ---- 2. high_monthly_charges_flag ----
    df["high_monthly_charges_flag"] = (df["MonthlyCharges"] > 80).astype(int)

    # ---- 3. payment_method_risk ----
    # High churn risk for electronic check
    df["payment_method_risk"] = df["PaymentMethod"].map({
        "Electronic check": 2,
        "Mailed check": 1,
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 0
    }).fillna(1)

    # ---- 4. contract_type_risk ----
    df["contract_type_risk"] = df["Contract"].map({
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0
    }).fillna(2)

    # ---- 5. tenure_bucket ----
    bins = [0, 12, 24, 48, 72, 999]
    labels = ["0-1", "1-2", "2-4", "4-6", "6+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    # Ensure it’s string
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)

    return df




def run_batch_prediction(df, model_path="models/best_churn_model.joblib"):
    """
    Runs prediction on the entire DataFrame.
    """
    model = load_model(model_path)

    processed = preprocess_batch(df)

    prob = model.predict_proba(processed)[:, 1]
    pred = (prob >= 0.5).astype(int)

    results = df.copy()
    results["churn_probability"] = prob.round(4)
    results["churn_prediction"] = pred

    return results
