#!/bin/bash
# This script starts the Gunicorn server for the Kairn application.

# Set the number of worker processes
WORKERS=4

# Set the host and port to bind to
BIND_ADDR=0.0.0.0:8000

# Change to the backend directory where the application and static files are located
cd backend

# Start Gunicorn with Uvicorn workers
# This is a production-ready command.
echo "Starting Kairn server with Gunicorn..."
echo "Starting Kairn server with Gunicorn..."
gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker -b $BIND_ADDR main:app
gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker -b $BIND_ADDR main:app
