
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"


def load_telco_data(path: str) -> pd.DataFrame:
    """Load Telco Customer Churn csv file."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Handle basic cleaning: TotalCharges type + missing values."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Map Churn Yes/No to 1/0 and return X, y."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected '{TARGET_COL}' column in data.")
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-style engineered features."""
    df = df.copy()

    # Tenure bucket
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1 yr", "1-2 yrs", "2-4 yrs", "4-6 yrs"],
            include_lowest=True,
        )

    # Ratio and flags
    if {"MonthlyCharges", "TotalCharges"}.issubset(df.columns):
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (
            df["TotalCharges"].replace(0, 1)
        )
        df["high_monthly_charges_flag"] = (
            df["MonthlyCharges"] > df["MonthlyCharges"].median()
        ).astype(int)

    # Payment method risk
    if "PaymentMethod" in df.columns:
        risky_methods = ["Mailed check", "Electronic check"]
        df["payment_method_risk"] = df["PaymentMethod"].isin(risky_methods).astype(int)

    # Contract type risk
    if "Contract" in df.columns:
        df["contract_type_risk"] = df["Contract"].apply(
            lambda x: 1 if x == "Month-to-month" else 0
        )

    return df


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical columns."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocess_transformer(df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features."""
    num_cols, cat_cols = get_feature_types(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor
