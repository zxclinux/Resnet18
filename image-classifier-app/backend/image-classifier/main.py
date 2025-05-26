from fastapi import FastAPI
from views import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CIFAR-10 Classifier API",
    description="API for images classification ResNet18",
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React у розробці
        "http://frontend:3000"    # React у Docker-мережі
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)