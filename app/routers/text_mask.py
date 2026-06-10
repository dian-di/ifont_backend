import io

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import Response

from app.utils.text_mask_sam import extract_handwritten_text

router = APIRouter(prefix="/text-mask", tags=["text-mask"])


@router.post(
    "/extract",
    summary="提取图片中的文字掩膜",
    description="上传一张图片，返回带透明通道的 PNG（仅保留文字区域）。设置 debug_dir 可将中间步骤图片保存到指定目录。",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "处理后的 RGBA PNG 图片"},
        400: {"description": "文件为空或不是有效图片"},
        413: {"description": "文件过大"},
    },
)
async def extract_mask(
    file: UploadFile = File(..., description="待处理的图片文件（JPEG/PNG/BMP/TIFF）"),
    debug_dir: str | None = Query(
        None,
        description="调试目录路径，设置后会将每步中间图片保存到该目录（如 /tmp/debug）",
    ),
) -> Response:
    image_bytes = await file.read()

    if not image_bytes:
        return Response(
            content="文件为空".encode(),
            status_code=400,
            media_type="text/plain",
        )

    if len(image_bytes) > 20 * 1024 * 1024:
        return Response(
            content="文件过大，限制 20MB".encode(),
            status_code=413,
            media_type="text/plain",
        )

    try:
        result_image = extract_handwritten_text(image_bytes, debug_dir=debug_dir)
    except ValueError as exc:
        return Response(
            content=str(exc).encode(),
            status_code=400,
            media_type="text/plain",
        )

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
