import pandas as pd
import pickle

# Load scaler
scaler = pickle.load(open("app/model/scaler.pkl", "rb"))

# EXACT columns (same as training)
columns = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

def preprocess(data):
    # Convert input to DataFrame
    df = pd.DataFrame([data])
    # Ensure correct order
    df = df[columns]

    # Scale Amount
    df["Amount"] = scaler.transform(df[["Amount"]])

    return df