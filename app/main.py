from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.routes.predict import router

app = FastAPI(
    title="Fraud Detection API",
    description="ML API with proper imbalance handling",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API routes
app.include_router(router)

# Templates setup
templates = Jinja2Templates(directory="app/templates")

# 🔥 NEW: Frontend UI route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check
@app.get("/health")
def health():
    return {"status": "OK"}