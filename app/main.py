from fastapi import FastAPI

from app.routers import text_mask as text_mask_router

app = FastAPI(
    title="iFont Text Mask API",
    description="从图片中提取文字掩膜，去除非文字区域，保留透明通道 PNG。",
    version="0.1.0",
)

app.include_router(text_mask_router.router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
