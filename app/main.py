from fastapi import FastAPI
from app.routes.predict import router

app = FastAPI(
    title="Fraud Detection API",
    description="ML API with proper imbalance handling",
    version="2.0",
    docs_url="/docs",        
    redoc_url="/redoc"
)

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Fraud Detection API running 🚀"}

@app.get("/health")
def health():
    return {"status": "OK"}