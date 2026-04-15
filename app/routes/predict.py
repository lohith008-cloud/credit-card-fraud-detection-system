from fastapi import APIRouter
import pickle

from app.schemas.input_schema import Transaction
from app.utils.preprocess import preprocess
from app.utils.logger import log_prediction  # ✅ logging

router = APIRouter()

# Load model
model = pickle.load(open("app/model/model.pkl", "rb"))

# Load threshold
with open("app/model/threshold.txt", "r") as f:
    threshold = float(f.read())


@router.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input → dict → preprocess
        data = transaction.dict()
        processed = preprocess(data)

        # Predict probability
        prob = model.predict_proba(processed)[0][1]

        # Apply threshold
        prediction = int(prob >= threshold)

        # 🔥 Risk levels (production-style)
        if prob >= 0.8:
            risk = "HIGH"
        elif prob >= 0.5:
            risk = "MEDIUM"
        elif prob >= threshold:
            risk = "LOW"
        else:
            risk = "SAFE"

        # ✅ Log prediction (important for real-world)
        log_prediction(data, prob, prediction)

        # Final response
        return {
            "fraud": bool(prediction),
            "probability": float(prob),
            "threshold_used": threshold,
            "risk_level": risk,
            "message": "Fraud Detected 🚨" if prediction else "Safe Transaction ✅"
        }

    except Exception as e:
        return {
            "error": str(e)
        }