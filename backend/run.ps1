Write-Host "Starting CareScribe API Server..." -ForegroundColor Green
Set-Location $PSScriptRoot
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

