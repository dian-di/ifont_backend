"""
Handwritten text extraction using PaddleOCR + BiRefNet + CLAHE + ConnectedComponent filtering.

Pipeline: Image -> OCR Service -> Text Regions -> CLAHE Enhancement -> Matting Service -> RGBA
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# ---------------------------------------------------------------------------
# OCR Service
# ---------------------------------------------------------------------------


class OCRService:
    """PaddleOCR-based text detection and recognition."""

    def __init__(self, lang: str = "ch", use_angle_cls: bool = True):
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def detect(self, img_rgb: np.ndarray) -> list[dict]:
        """
        Detect text regions in an RGB image.

        Returns:
            List of dicts with keys: text, bbox (x_min, y_min, x_max, y_max), image (RGB crop)
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        result = self._ocr.ocr(img_bgr)
        if not result or not result[0]:
            return []

        ocr_data = result[0]
        regions: list[dict] = []

        def _append(poly, text: str):
            x1, y1, x2, y2 = _bbox_from_poly(poly)
            h, w = img_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return
            regions.append(
                {
                    "text": text,
                    "bbox": (x1, y1, x2, y2),
                    "image": img_rgb[y1:y2, x1:x2].copy(),
                }
            )

        if isinstance(ocr_data, dict):
            dt_polys = ocr_data.get("dt_polys", [])
            rec_texts = ocr_data.get("rec_texts", [])
            for i, poly in enumerate(dt_polys):
                _append(poly, rec_texts[i] if i < len(rec_texts) else "")
        elif hasattr(ocr_data, "dt_polys"):
            dt_polys = ocr_data.dt_polys
            rec_texts = getattr(ocr_data, "rec_texts", [])
            for i, poly in enumerate(dt_polys):
                _append(poly, rec_texts[i] if i < len(rec_texts) else "")
        else:
            for line in ocr_data:
                _append(line[0], line[1][0])

        return regions


def _bbox_from_poly(poly) -> tuple[int, int, int, int]:
    """Convert polygon points to axis-aligned bounding box (x1, y1, x2, y2)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# ---------------------------------------------------------------------------
# Matting Service (BiRefNet)
# ---------------------------------------------------------------------------


class MattingService:
    """BiRefNet-based image matting for precise text foreground extraction."""

    MODEL_NAME = "ZhengPeng7/BiRefNet"

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    _CPU_INPUT_SIZE = 512
    _GPU_INPUT_SIZE = 1024

    def __init__(self, device: str | None = None):
        from transformers import AutoModelForImageSegmentation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.float()

        self.model.to(self.device)
        self.model.eval()

        input_size = self._CPU_INPUT_SIZE if self.device == "cpu" else self._GPU_INPUT_SIZE
        self._resize = transforms.Resize((input_size, input_size))
        self._dtype = next(self.model.parameters()).dtype

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """Build a normalised [1, 3, H, W] tensor from a PIL image."""
        pil_resized = self._resize(pil_img)
        arr = np.array(pil_resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(self._IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self._IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(device=self.device, dtype=self._dtype)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Generate alpha matte for an RGB image crop.

        Returns:
            HxW uint8 alpha matte (0 = background, 255 = foreground)
        """
        h, w = img_rgb.shape[:2]
        pil_img = Image.fromarray(img_rgb)
        tensor = self._preprocess(pil_img)

        with torch.no_grad():
            pred = self.model(tensor)[-1].sigmoid()

        alpha = pred[0].squeeze().cpu().numpy()
        alpha = (alpha * 255).astype(np.uint8)
        return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# CLAHE Enhancement
# ---------------------------------------------------------------------------


def apply_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE on the L channel of LAB color space to enhance text contrast.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Connected Component Filtering
# ---------------------------------------------------------------------------


def filter_connected_components(
    alpha: np.ndarray,
    min_area: int = 50,
    max_area_ratio: float = 0.5,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 15.0,
) -> np.ndarray:
    """
    Clean an alpha matte using connected-component analysis.
    """
    binary = (alpha > 127).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return alpha

    h, w = alpha.shape[:2]
    max_area = h * w * max_area_ratio
    cleaned = alpha.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = cw / ch if ch > 0 else 0

        if area < min_area or area > max_area:
            cleaned[labels == i] = 0
        elif aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            cleaned[labels == i] = 0

    return cleaned


# ---------------------------------------------------------------------------
# Debug visualisation helpers
# ---------------------------------------------------------------------------


def _annotate_ocr_bbox(img_rgb: np.ndarray, regions: list[dict]) -> Image.Image:
    """Draw bounding boxes + labels on a copy of the original image."""
    canvas = img_rgb.copy()
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"[{i}] {r['text'][:15]}"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


def _crops_montage(
    rgb_crops: list[np.ndarray],
    labels: list[str] | None = None,
    max_h: int = 200,
) -> Image.Image:
    """Place all crops in a horizontal strip, auto-resized to *max_h*."""
    if not rgb_crops:
        return Image.new("RGB", (400, 60), (200, 200, 200))

    strips: list[np.ndarray] = []
    for idx, crop in enumerate(rgb_crops):
        if crop is None or crop.size == 0:
            continue
        if crop.ndim == 2:                       # grayscale → RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        h, w = crop.shape[:2]
        scale = max_h / h if h > max_h else 1.0
        resized = cv2.resize(crop, (max(int(w * scale), 1), max_h))
        # optional label bar on top
        if labels and idx < len(labels):
            bar = np.full((20, resized.shape[1], 3), 40, dtype=np.uint8)
            cv2.putText(bar, labels[idx], (2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            resized = np.vstack([bar, resized])
        strips.append(resized)

    if not strips:
        return Image.new("RGB", (400, 60), (200, 200, 200))

    spacing = 6
    total_w = sum(s.shape[1] for s in strips) + spacing * (len(strips) - 1)
    canvas_h = max(s.shape[0] for s in strips)
    canvas = np.full((canvas_h, total_w, 3), 220, dtype=np.uint8)
    x = 0
    for s in strips:
        canvas[: s.shape[0], x : x + s.shape[1]] = s
        x += s.shape[1] + spacing
    return Image.fromarray(canvas)


def _alpha_montage(
    alphas: list[np.ndarray],
    labels: list[str] | None = None,
    max_h: int = 200,
) -> Image.Image:
    """Visualise alpha mattes as grayscale strips (0=black, 255=white)."""
    rgb_list = []
    for a in alphas:
        if a is None or a.size == 0:
            rgb_list.append(np.zeros((10, 10, 3), dtype=np.uint8))
        else:
            rgb_list.append(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
    return _crops_montage(rgb_list, labels=labels, max_h=max_h)


# ---------------------------------------------------------------------------
# Lazy singleton services
# ---------------------------------------------------------------------------

_ocr_service: OCRService | None = None
_matting_service: MattingService | None = None


def _get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(lang="ch", use_angle_cls=True)
    return _ocr_service


def _get_matting_service() -> MattingService:
    global _matting_service
    if _matting_service is None:
        _matting_service = MattingService()
    return _matting_service


# ---------------------------------------------------------------------------
# Debug file writer
# ---------------------------------------------------------------------------

_DEBUG_STEPS = [
    "01_original",
    "02_ocr_bbox",
    "03_ocr_crops",
    "04_clahe",
    "05_matting_alpha",
    "06_cc_filtered",
    "07_final_rgba",
]


def _save_debug_images(debug_dir: str, images: dict[str, Image.Image]) -> None:
    """Write every intermediate image to *debug_dir* as PNG."""
    os.makedirs(debug_dir, exist_ok=True)
    for name, img in images.items():
        img.save(os.path.join(debug_dir, f"{name}.png"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_handwritten_text(
    image_bytes: bytes,
    output_path: str | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
    debug_dir: str | None = None,
) -> Image.Image:
    """
    Extract handwritten text with precise foreground matting.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG/BMP/TIFF)
        output_path: Optional path to save the final RGBA PNG
        background_color: Unused in RGBA mode (kept for API compat)
        debug_dir: If set, all intermediate step images are saved here as PNGs:
                   01_original, 02_ocr_bbox, 03_ocr_crops, 04_clahe,
                   05_matting_alpha, 06_cc_filtered, 07_final_rgba

    Returns:
        PIL Image in RGBA mode (transparent background)
    """
    debug: dict[str, Image.Image] = {}

    # ---- 1. Decode ----
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("无法解码图片，请确认文件格式正确")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    debug["original"] = Image.fromarray(img_rgb)

    # ---- 2. OCR ----
    ocr_service = _get_ocr_service()
    text_regions = ocr_service.detect(img_rgb)

    debug["ocr_bbox"] = _annotate_ocr_bbox(img_rgb, text_regions)

    labels = [f"[{i}] {r.get('text', '')[:8]}" for i, r in enumerate(text_regions)]
    debug["ocr_crops"] = _crops_montage(
        [r["image"] for r in text_regions], labels=labels,
    )

    if not text_regions:
        empty = Image.new("RGBA", (img_rgb.shape[1], img_rgb.shape[0]), (0, 0, 0, 0))
        debug["clahe"] = empty
        debug["matting_alpha"] = empty
        debug["cc_filtered"] = empty
        debug["final_rgba"] = empty
        if debug_dir:
            _save_debug_images(debug_dir, debug)
        if output_path:
            empty.save(output_path)
        return empty

    # ---- 3. Per-region: CLAHE → BiRefNet → CC filter ----
    matting_service = _get_matting_service()

    clahe_crops: list[np.ndarray] = []
    raw_alphas: list[np.ndarray] = []
    filtered_alphas: list[np.ndarray] = []

    for region in text_regions:
        crop = region["image"]
        if crop.size == 0:
            region["alpha"] = np.zeros((0, 0), dtype=np.uint8)
            clahe_crops.append(crop)
            raw_alphas.append(np.zeros((10, 10), dtype=np.uint8))
            filtered_alphas.append(np.zeros((10, 10), dtype=np.uint8))
            continue

        # 3a. CLAHE
        enhanced = apply_clahe(crop)
        clahe_crops.append(enhanced)

        # 3b. BiRefNet matting
        alpha = matting_service.predict(enhanced)
        raw_alphas.append(alpha.copy())

        # 3c. ConnectedComponent filter
        alpha = filter_connected_components(alpha)
        filtered_alphas.append(alpha)

        region["alpha"] = alpha

    debug["clahe"] = _crops_montage(clahe_crops, labels=labels)
    debug["matting_alpha"] = _alpha_montage(raw_alphas, labels=labels)
    debug["cc_filtered"] = _alpha_montage(filtered_alphas, labels=labels)

    # ---- 4. Compose final RGBA ----
    lines = group_regions_into_lines(text_regions)
    output_img = compose_rgba_output(lines, img_rgb.shape[:2])
    debug["final_rgba"] = output_img

    if debug_dir:
        _save_debug_images(debug_dir, debug)
    if output_path:
        output_img.save(output_path)

    return output_img


# ---------------------------------------------------------------------------
# Line grouping & RGBA composition
# ---------------------------------------------------------------------------


def group_regions_into_lines(
    regions: list[dict],
    y_threshold: int = 30,
) -> list[list[dict]]:
    """Group detected text regions into horizontal lines by Y-coordinate proximity."""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r["bbox"][1])
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        prev = current_line[-1]
        if abs(region["bbox"][1] - prev["bbox"][1]) < y_threshold:
            current_line.append(region)
        else:
            current_line.sort(key=lambda r: r["bbox"][0])
            lines.append(current_line)
            current_line = [region]

    current_line.sort(key=lambda r: r["bbox"][0])
    lines.append(current_line)
    return lines


def compose_rgba_output(
    lines: list[list[dict]],
    original_shape: tuple[int, int],
    padding: int = 40,
    line_spacing: int = 20,
    word_spacing: int = 10,
) -> Image.Image:
    """
    Compose matted text regions into a single RGBA image.
    """
    if not lines:
        return Image.new("RGBA", (original_shape[1], original_shape[0]), (0, 0, 0, 0))

    line_meta: list[dict] = []
    max_line_width = 0
    total_height = 0

    for line in lines:
        line_w = 0
        line_h = 0
        for region in line:
            x1, y1, x2, y2 = region["bbox"]
            line_w += x2 - x1
            line_h = max(line_h, y2 - y1)
        if len(line) > 1:
            line_w += word_spacing * (len(line) - 1)
        max_line_width = max(max_line_width, line_w)
        total_height += line_h + line_spacing
        line_meta.append({"regions": line, "height": line_h})

    total_height -= line_spacing

    canvas_w = max_line_width + padding * 2
    canvas_h = total_height + padding * 2
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    current_y = padding
    for meta in line_meta:
        current_x = padding
        for region in meta["regions"]:
            crop_rgb = region["image"]
            alpha = region.get("alpha")
            if crop_rgb.size == 0 or alpha is None or alpha.size == 0:
                continue

            h, w = crop_rgb.shape[:2]
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

            end_y = min(current_y + h, canvas_h)
            end_x = min(current_x + w, canvas_w)
            paste_h = end_y - current_y
            paste_w = end_x - current_x
            if paste_h <= 0 or paste_w <= 0:
                current_x += w + word_spacing
                continue

            src_rgba = np.zeros((paste_h, paste_w, 4), dtype=np.uint8)
            src_rgba[:, :, :3] = crop_rgb[:paste_h, :paste_w]
            src_rgba[:, :, 3] = alpha[:paste_h, :paste_w]

            src_a = src_rgba[:, :, 3:4].astype(np.float32) / 255.0
            dst_region = canvas[current_y:end_y, current_x:end_x].astype(np.float32) / 255.0
            dst_a = dst_region[:, :, 3:4]

            out_a = src_a + dst_a * (1.0 - src_a)
            out_rgb = (src_rgba[:, :, :3].astype(np.float32) / 255.0 * src_a
                       + dst_region[:, :, :3] * dst_a * (1.0 - src_a))

            composite = np.zeros((paste_h, paste_w, 4), dtype=np.float32)
            safe_a = np.where(out_a > 0, out_a, 1.0)
            composite[:, :, :3] = out_rgb / safe_a
            composite[:, :, 3] = out_a[:, :, 0]

            canvas[current_y:end_y, current_x:end_x] = np.clip(
                composite * 255, 0, 255
            ).astype(np.uint8)

            current_x += w + word_spacing

        current_y += meta["height"] + line_spacing

    return Image.fromarray(canvas, "RGBA")
