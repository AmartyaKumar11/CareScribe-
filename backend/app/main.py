from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import sessions, audio

app = FastAPI(title="CareScribe API", version="1.0.0")

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(audio.router, prefix="/sessions", tags=["audio"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

