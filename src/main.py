from fastapi import FastAPI
from routes_files import router as files_router


app = FastAPI()

# mount the router
app.include_router(files_router, prefix="/api", tags=["files"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
