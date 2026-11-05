"""
Voynich Vision Segmenter
Module for preprocessing manuscript page images, isolating parchment noise,
detecting text vs illustrations, and extracting lines and word bounding boxes.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Safely load image from disk and return BGR image array.
    If file is missing or unreadable, creates and returns a parchment placeholder array.
    """
    import os
    if not image_path or not os.path.isfile(image_path):
        h, w = 1000, 750
        fallback = np.full((h, w, 3), (205, 222, 235), dtype=np.uint8)
        cv2.rectangle(fallback, (15, 15), (w - 15, h - 15), (140, 160, 180), 2)
        text = os.path.basename(image_path) if image_path else "Bilinmeyen"
        cv2.putText(fallback, f"Gorsel Bulunamadi: {text}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 40, 30), 2)
        return fallback

    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            h, w = 1000, 750
            img = np.full((h, w, 3), (205, 222, 235), dtype=np.uint8)
    return img


def binarize_page(img: np.ndarray) -> np.ndarray:
    """
    Preprocess manuscript page image:
    1. Grayscale conversion
    2. Gaussian filtering to suppress parchment texture
    3. Adaptive thresholding where foreground (ink) is 255 and background is 0.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    bin_inv = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    return bin_inv


def filter_text_strokes(bin_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Filters out tiny speckles and large illustration blobs, preserving letter strokes.
    Returns clean binary image and list of stroke bounding boxes (x, y, w, h).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    if num_labels <= 1:
        return np.zeros_like(bin_img), []

    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (h >= 6) & (h <= 95) & (w >= 3) & (w <= 85) & (area >= 12) & (area <= 2500)
    keep[0] = False  # background

    lut = np.where(keep, 255, 0).astype(np.uint8)
    clean_text = lut[labels]
    strokes = [(int(stats[i, 0]), int(stats[i, 1]), int(w[i]), int(h[i])) for i in np.where(keep)[0]]

    return clean_text, strokes



def segment_words_and_lines(clean_text: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Groups strokes into words and lines using horizontal morphological structuring.
    Returns:
    - line_boxes: list of (x, y, w, h)
    - word_boxes: list of (x, y, w, h)
    """
    h_img, w_img = clean_text.shape

    # 1. Word level grouping (small horizontal kernel)
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
    word_dilated = cv2.dilate(clean_text, word_kernel, iterations=1)
    w_contours, _ = cv2.findContours(word_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_boxes = []
    for cnt in w_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw >= 10 and bh >= 8:
            word_boxes.append((bx, by, bw, bh))

    word_boxes.sort(key=lambda b: (b[1] // 25, b[0]))

    # 2. Line level grouping (wider horizontal kernel)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 4))
    line_dilated = cv2.dilate(clean_text, line_kernel, iterations=2)
    l_contours, _ = cv2.findContours(line_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_boxes = []
    for cnt in l_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw >= 30 and bh >= 12:
            line_boxes.append((bx, by, bw, bh))

    line_boxes.sort(key=lambda b: b[1])

    return line_boxes, word_boxes


def segment_lines(bin_img: np.ndarray, **kwargs) -> List[Tuple[int, int, int, int]]:
    """Convenience wrapper for line segmentation."""
    clean_text, _ = filter_text_strokes(bin_img)
    lines, _ = segment_words_and_lines(clean_text)
    return lines


def process_page(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline to process a page image and return structured segmentation data.
    """
    orig_bgr = load_image(image_path)
    bin_img = binarize_page(orig_bgr)
    clean_text, strokes = filter_text_strokes(bin_img)
    line_boxes, word_boxes = segment_words_and_lines(clean_text)

    all_words = []
    for w_idx, wb in enumerate(word_boxes):
        matching_line = 0
        for l_idx, lb in enumerate(line_boxes):
            if lb[1] <= wb[1] <= lb[1] + lb[3] or abs(wb[1] - lb[1]) < 20:
                matching_line = l_idx
                break
        all_words.append({
            'line_idx': matching_line,
            'word_idx': w_idx,
            'bbox': wb
        })

    return {
        'image_shape': orig_bgr.shape,
        'line_boxes': line_boxes,
        'words': all_words,
        'line_count': len(line_boxes),
        'word_count': len(all_words),
        'stroke_count': len(strokes)
    }


def draw_segmentation_overlay(img: np.ndarray, seg_result: Dict[str, Any]) -> np.ndarray:
    """
    Draw colored bounding boxes on the original image for lines (blue) and words (green).
    """
    overlay = img.copy()
    
    for (x, y, w, h) in seg_result['line_boxes']:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 120, 0), 2)

    for word in seg_result['words']:
        (x, y, w, h) = word['bbox']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 1)

    return overlay
