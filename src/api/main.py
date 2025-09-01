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
