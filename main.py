from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.facial_auth import router as facial_auth_router
from config import Config

app = FastAPI(
    title="Facial Authentication System",
    description="Standalone facial recognition system for Accessly",
    version="1.0.0",
    max_upload_size=10 * 1024 * 1024  # 10MB
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX: Change the prefix to match what Node.js is calling
app.include_router(
    facial_auth_router, 
    prefix="/api",  # Changed from "/api/v1/facial-auth"
    tags=["Facial Authentication"]
)

@app.get("/")
async def root():
    return {"message": "Facial Authentication System is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "facial-auth"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )