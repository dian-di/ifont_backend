import io

import cv2
import numpy as np
from fastapi import FastAPI, File, Response, UploadFile
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI(title="PaddleOCR Text Extraction")

# ================= 1. 模型初始化 =================
# 使用 PaddleOCR 轻量级模型，仅检测 (det=True, rec=False)
# use_gpu=False 即可在 CPU 上流畅运行，速度极快
ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False, use_gpu=False)


@app.post("/extract_paddle")
async def extract_paddle(file: UploadFile = File(...)):
    try:
        # 1. 读取图片
        img_bytes = await file.read()
        img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. PaddleOCR 检测文字框
        # ocr_results 格式: [[[box_points], (text, prob)], ...]
        ocr_results = ocr.ocr(img_bgr, cls=False)

        if not ocr_results or not ocr_results[0]:
            return Response(content=img_bytes, media_type=file.content_type)

        # 3. 创建最终 Mask (全黑)
        final_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        # 4. 遍历每个文字框，在框内进行精准颜色提取
        for line in ocr_results[0]:
            box_points = np.array(line[0], dtype=np.int32)  # 形状: [4, 2]

            # 获取 Box 的外接矩形 (x, y, w, h)
            x, y, w, h = cv2.boundingRect(box_points)

            # 裁剪出 Box 区域 (ROI)
            roi_bgr = img_bgr[y : y + h, x : x + w]
            roi_rgb = img_np[y : y + h, x : x + w]

            # 在 ROI 内进行 HSV 红色提取 (避开外部干扰)
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            # 红色阈值 (根据图1调整)
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 80, 80])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
            roi_mask = mask1 + mask2

            # 可选：形态学操作，去除框内细小的噪点
            kernel = np.ones((2, 2), np.uint8)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

            # 将 ROI 的 Mask 贴回最终 Mask 的对应位置
            final_mask[y : y + h, x : x + w] = np.maximum(
                final_mask[y : y + h, x : x + w], roi_mask
            )

        # 5. 图像合成
        bg_color = np.array([240, 240, 240], dtype=np.uint8)
        mask_3d = np.stack([final_mask] * 3, axis=-1)

        # 保留文字像素，其余换背景
        result_img = np.where(mask_3d > 0, img_np, bg_color)

        # 6. 返回
        result_pil = Image.fromarray(result_img)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
