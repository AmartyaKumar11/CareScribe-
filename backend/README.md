# CareScribe Backend

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

### Option 1: Using Python module (Recommended)
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### Option 2: Using uvicorn directly (if in PATH)
```bash
cd backend
uvicorn app.main:app --reload
```

### Option 3: Using the run script (Windows)
```bash
# PowerShell
.\run.ps1

# Command Prompt
run.bat
```

The server will start at `http://localhost:8000`

## API Endpoints

- `GET /health` - Health check
- `POST /sessions` - Create a new session
- `POST /sessions/{session_id}/audio` - Upload audio file
- `GET /sessions/{session_id}/transcript` - Get transcript (stubbed)

## Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

