"""
api/main.py

This module initializes and runs the FastAPI application for the document
management and retrieval-augmented generation (RAG) service. This module uses routes
defined in `src.api.routes` to handle API requests.

Key functionalities:
1. Creates a FastAPI app instance.
2. Mounts the API router defined in `src.api.routes` under the `/api` prefix,
   with the "files" tag for documentation grouping.
3. Runs the application using Uvicorn when executed as the main program.

Usage:
    Run this module directly to start the server:
        python main.py
    The API will be accessible at http://0.0.0.0:8001/api
"""


from fastapi import FastAPI
from src.api.routes import router 


app = FastAPI()

# mount the router
app.include_router(router, prefix="/api", tags=["files"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
