# Build the frontend and start the server for the class demo (CLAUDE.md §8).
# Serves the whole app (UI + API + WebSocket) from http://localhost:8000, offline.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== Building frontend (web/dist) ==" -ForegroundColor Cyan
Push-Location web
npm run build
Pop-Location

Write-Host "== Starting Shock Lab server at http://localhost:8000 ==" -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m shocklab.server --port 8000
