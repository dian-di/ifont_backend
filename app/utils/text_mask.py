import cv2
import numpy as np
from PIL import Image
import os

def extract_text_mask_opencv(image_path, output_path=None, debug=False):
    """
    从路边招牌图像中提取文字掩膜（保留原始字体字色）
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（支持PNG，带透明通道）
        debug: 是否显示中间处理步骤
    
    Returns:
        result_rgba: 带透明通道的文字图像 (RGBA格式)
        mask: 二值掩膜
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # 2. 提取亮度通道 L
    L_channel = img_lab[:, :, 0]
    
    if debug:
        cv2.imshow("Step 1: L Channel", L_channel)
    
    # 3. 自适应阈值分割（核心步骤）
    # 方法1：高斯自适应阈值（推荐）
    mask = cv2.adaptiveThreshold(
        L_channel, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 文字暗背景亮用INV
        blockSize=25,           # 局部窗口大小（奇数，越大越抗噪）
        C=8                     # 偏置值（正数减少误检）
    )
    
    if debug:
        cv2.imshow("Step 2: Adaptive Threshold", mask)
    
    # 可选：尝试另一种方法（均值自适应阈值）
    # mask = cv2.adaptiveThreshold(L_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    
    # 4. 形态学处理 - 连接断裂笔画
    # 根据文字方向选择结构元素（横排文字用水平椭圆）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 闭运算：先膨胀后腐蚀，连接断裂区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 轻微开运算：去除小噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    if debug:
        cv2.imshow("Step 3: Morphological Close + Open", mask)
    
    # 5. 连通域分析 - 过滤噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 计算图像面积
    img_area = mask.shape[0] * mask.shape[1]
    
    # 过滤参数（根据实际图像调整）
    min_area = 50                    # 最小文字面积
    max_area = img_area * 0.3        # 最大文字面积（避免整图被识别）
    min_aspect_ratio = 0.2           # 最小宽高比（太细长的可能是线条）
    max_aspect_ratio = 10.0          # 最大宽高比
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height if height > 0 else 0
        
        # 过滤条件
        if area < min_area or area > max_area:
            mask[labels == i] = 0
        elif aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            mask[labels == i] = 0
        # 可选：过滤过于接近边界的区域
        # x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        # if x < 5 or y < 5 or x + width > mask.shape[1] - 5 or y + height > mask.shape[0] - 5:
        #     mask[labels == i] = 0
    
    if debug:
        cv2.imshow("Step 4: Connected Components Filter", mask)
    
    # 6. 可选：边缘平滑（使用中值滤波）
    mask = cv2.medianBlur(mask, 3)
    
    if debug:
        cv2.imshow("Step 5: Final Mask", mask)
    
    # 7. 生成带透明通道的结果图像
    # 创建RGBA图像
    height, width = img_rgb.shape[:2]
    result_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 复制RGB通道（仅在掩膜区域）
    result_rgba[:, :, 0:3] = img_rgb
    
    # 设置Alpha通道（掩膜区域不透明，背景透明）
    result_rgba[:, :, 3] = mask
    
    # 8. 保存结果
    if output_path:
        # 确保输出路径是PNG格式（支持透明通道）
        if not output_path.endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        
        # 使用PIL保存（cv2.imwrite不支持4通道PNG？实际上支持，但为了保险用PIL）
        result_pil = Image.fromarray(result_rgba, 'RGBA')
        result_pil.save(output_path)
        print(f"结果已保存: {output_path}")
    
    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_rgba, mask


def extract_text_mask_advanced(image_path, output_path=None, debug=False):
    """
    增强版：结合多种策略处理不同光照和背景情况
    适用于复杂背景（如木头纹理、彩色背景等）
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 策略1：尝试Lab空间的L通道
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = img_lab[:, :, 0]
    
    # 策略2：尝试HSV空间的V通道
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = img_hsv[:, :, 2]
    
    # 策略3：计算梯度（适用于边缘明显的文字）
    grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = np.uint8(np.clip(gradient, 0, 255))
    
    if debug:
        cv2.imshow("L Channel", L)
        cv2.imshow("V Channel", V)
        cv2.imshow("Gradient", gradient)
    
    # 选择最佳通道（这里简单用L通道，实际可以自动判断对比度）
    best_channel = L
    
    # 多阈值尝试
    masks = []
    block_sizes = [15, 25, 35]  # 不同窗口大小
    C_values = [5, 8, 12]       # 不同偏置值
    
    for block_size in block_sizes:
        for C in C_values:
            mask = cv2.adaptiveThreshold(
                best_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, C
            )
            masks.append(mask)
    
    # 合并掩膜（取并集）
    mask_combined = np.zeros_like(best_channel, dtype=np.uint8)
    for m in masks:
        mask_combined = cv2.bitwise_or(mask_combined, m)
    
    if debug:
        cv2.imshow("Combined Masks", mask_combined)
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 连通域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_combined, connectivity=8)
    min_area = 50
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            mask_combined[labels == i] = 0
    
    if debug:
        cv2.imshow("Final Mask Advanced", mask_combined)
    
    # 生成结果
    height, width = img_rgb.shape[:2]
    result_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    result_rgba[:, :, 0:3] = img_rgb
    result_rgba[:, :, 3] = mask_combined
    
    if output_path:
        if not output_path.endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        result_pil = Image.fromarray(result_rgba, 'RGBA')
        result_pil.save(output_path)
        print(f"结果已保存: {output_path}")
    
    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_rgba, mask_combined


def batch_process(input_dir, output_dir, use_advanced=False, debug=False):
    """
    批量处理目录下的所有图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        use_advanced: 是否使用增强版
        debug: 是否显示调试信息
    """
    os.makedirs(output_dir, exist_ok=True)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '_text.png'
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"处理: {filename}")
            
            try:
                if use_advanced:
                    extract_text_mask_advanced(input_path, output_path, debug=False)
                else:
                    extract_text_mask_opencv(input_path, output_path, debug=False)
                print(f"  -> 已保存: {output_path}")
            except Exception as e:
                print(f"  -> 处理失败: {e}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1：单张图片测试（显示调试过程）
    test_image = "test.jpg"  # 替换为您的图片路径
    
    # 如果测试图片存在
    if os.path.exists(test_image):
        print("=== 基础版测试 ===")
        result, mask = extract_text_mask_opencv(test_image, "output_basic.png", debug=True)
        
        print("\n=== 增强版测试 ===")
        result_adv, mask_adv = extract_text_mask_advanced(test_image, "output_advanced.png", debug=True)
    else:
        print(f"测试图片不存在: {test_image}")
        print("请将您的图片放在当前目录，命名为 test.jpg，或修改代码中的路径")
    
    # 示例2：批量处理
    # batch_process("./input_images", "./output_images", use_advanced=False)
    
    # 示例3：仅获取掩膜（不保存）
    # if os.path.exists(test_image):
    #     result_rgba, mask = extract_text_mask_opencv(test_image, debug=False)
    #     print(f"掩膜形状: {mask.shape}, 文字像素数: {np.sum(mask > 0)}")

def extract_text_mask_from_bytes(
    image_bytes: bytes,
    use_advanced: bool = False,
) -> bytes:
    """
    Accept raw image bytes, return PNG bytes with transparent background.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG/PNG/BMP/etc.)
        use_advanced: Use the advanced multi-strategy extraction

    Returns:
        PNG image bytes (RGBA, non-text areas transparent)
    """
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像，请检查上传的文件是否为有效图片")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    if use_advanced:
        # --- advanced strategy (mirrors extract_text_mask_advanced) ---
        L = img_lab[:, :, 0]
        masks = []
        for block_size in (15, 25, 35):
            for C in (5, 8, 12):
                masks.append(
                    cv2.adaptiveThreshold(
                        L, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        block_size, C,
                    )
                )
        mask = np.zeros_like(L, dtype=np.uint8)
        for m in masks:
            mask = cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50:
                mask[labels == i] = 0
    else:
        # --- basic strategy (mirrors extract_text_mask_opencv) ---
        L_channel = img_lab[:, :, 0]
        mask = cv2.adaptiveThreshold(
            L_channel, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25, C=8,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        img_area = mask.shape[0] * mask.shape[1]
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            ar = w / h if h > 0 else 0
            if area < 50 or area > img_area * 0.3:
                mask[labels == i] = 0
            elif ar < 0.2 or ar > 10.0:
                mask[labels == i] = 0

        mask = cv2.medianBlur(mask, 3)

    # Build RGBA result
    height, width = img_rgb.shape[:2]
    result_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    result_rgba[:, :, :3] = img_rgb
    result_rgba[:, :, 3] = mask

    # Encode to PNG bytes
    _, buf = cv2.imencode(".png", cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA))
    return buf.tobytes()
