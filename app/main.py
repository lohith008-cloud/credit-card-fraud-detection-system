from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from app.routes.predict import router

app = FastAPI(
    title="Fraud Detection API",
    description="ML API with proper imbalance handling",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        return templates.TemplateResponse(
            name="index.html",
            context={"request": request}
        )
    except Exception as e:
        return {"error": str(e)}   # 👈 THIS LINE

@app.get("/health")
def health():
    return {"status": "OK"}