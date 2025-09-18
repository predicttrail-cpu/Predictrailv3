#!/bin/bash
# This script starts the Uvicorn server for the Kairn application for local development.

# Change to the backend directory where the application is located
cd backend

# Start Uvicorn server with auto-reloading
echo "Starting Kairn server with Uvicorn for local development..."
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
