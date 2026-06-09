from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import Response

from app.utils.text_mask import extract_text_mask_from_bytes

router = APIRouter(prefix="/text-mask", tags=["text-mask"])


@router.post(
    "/extract",
    summary="提取图片中的文字掩膜",
    description="上传一张图片，返回带透明通道的 PNG（仅保留文字区域）。",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "处理后的 RGBA PNG 图片"},
        400: {"description": "文件为空或不是有效图片"},
        413: {"description": "文件过大"},
    },
)
async def extract_mask(
    file: UploadFile = File(..., description="待处理的图片文件（JPEG/PNG/BMP/TIFF）"),
    advanced: bool = Query(False, description="是否使用增强版算法（适用于复杂背景）"),
) -> Response:
    image_bytes = await file.read()

    if not image_bytes:
        return Response(
            content="文件为空".encode(), status_code=400, media_type="text/plain",
        )

    if len(image_bytes) > 20 * 1024 * 1024:
        return Response(
            content="文件过大，限制 20MB".encode(), status_code=413, media_type="text/plain",
        )

    try:
        png_bytes = extract_text_mask_from_bytes(image_bytes, use_advanced=advanced)
    except ValueError as exc:
        return Response(
            content=str(exc).encode(), status_code=400, media_type="text/plain",
        )

    return Response(content=png_bytes, media_type="image/png")
