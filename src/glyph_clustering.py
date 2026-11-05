"""
Voynich Glyph Clustering (Unsupervised OCR)
Extracts glyph components from words, computes shape feature vectors,
and clusters them without prior alphabet knowledge to discover natural grapheme classes.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def extract_glyphs_from_word(bin_img: np.ndarray, word_bbox: Tuple[int, int, int, int], min_area: int = 12) -> List[Dict[str, Any]]:
    """
    Isolate individual glyphs inside a word bounding box using connected components.
    """
    x, y, w, h = word_bbox
    # Boundary check
    img_h, img_w = bin_img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return []

    word_roi = bin_img[y1:y2, x1:x2]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_roi, connectivity=8)
    glyphs = []

    for i in range(1, num_labels):
        gx, gy, gw, gh, area = stats[i]
        if area >= min_area and gw >= 3 and gh >= 5:
            glyph_crop = word_roi[gy:gy+gh, gx:gx+gw]
            glyphs.append({
                'rel_bbox': (gx, gy, gw, gh),
                'abs_bbox': (x1 + gx, y1 + gy, gw, gh),
                'area': area,
                'aspect_ratio': gw / max(1, gh),
                'image': glyph_crop
            })

    # Sort glyphs left to right
    glyphs.sort(key=lambda g: g['abs_bbox'][0])
    return glyphs


def extract_features(glyph_img: np.ndarray, target_size: Tuple[int, int] = (24, 24)) -> np.ndarray:
    """
    Normalize glyph crop to target_size and extract spatial grid density and contour moments.
    """
    resized = cv2.resize(glyph_img.astype(np.float32), target_size, interpolation=cv2.INTER_AREA)
    norm = resized / 255.0

    # 1. 4x4 spatial grid density (16 features)
    cells_y, cells_x = 4, 4
    h_step, w_step = target_size[1] // cells_y, target_size[0] // cells_x
    spatial_features = []
    for r in range(cells_y):
        for c in range(cells_x):
            cell = norm[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            spatial_features.append(np.mean(cell))

    # 2. Horizontal and vertical projection profiles (normalized)
    h_proj = np.sum(norm, axis=1) / (np.sum(norm) + 1e-6)
    v_proj = np.sum(norm, axis=0) / (np.sum(norm) + 1e-6)

    # 3. Geometric features
    moments = cv2.moments(norm)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    feat_vector = np.concatenate([spatial_features, h_proj, v_proj, hu_log])
    return feat_vector


def cluster_glyphs(glyphs: List[Dict[str, Any]], n_clusters: int = 15) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
    """
    Cluster extracted glyphs into discrete grapheme classes.
    Gracefully handles any number of glyphs.
    """
    n_samples = len(glyphs)
    if n_samples == 0:
        return np.array([]), {}, np.zeros((0, 2))

    if n_samples == 1:
        return np.array([0]), {0: [0]}, np.zeros((1, 2))

    effective_k = min(n_clusters, n_samples)
    effective_k = max(2, effective_k)
    if effective_k > n_samples:
        effective_k = n_samples

    features = np.array([extract_features(g['image']) for g in glyphs])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if effective_k == n_samples:
        labels = np.arange(n_samples)
    else:
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)

    # 2D PCA projection for inspection
    if n_samples >= 2:
        pca = PCA(n_components=min(2, features.shape[1], n_samples))
        pca_proj = pca.fit_transform(features)
        if pca_proj.shape[1] == 1:
            pca_2d = np.hstack([pca_proj, np.zeros((n_samples, 1))])
        else:
            pca_2d = pca_proj
    else:
        pca_2d = np.zeros((n_samples, 2))

    cluster_indices = {}
    for idx, label in enumerate(labels):
        cluster_indices.setdefault(int(label), []).append(idx)

    return labels, cluster_indices, pca_2d


def compute_cluster_representatives(glyphs: List[Dict[str, Any]], cluster_indices: Dict[int, List[int]], target_size: Tuple[int, int] = (28, 28)) -> Dict[int, np.ndarray]:
    """
    Average images of glyphs in each cluster to produce a synthetic average glyph visual.
    """
    reps = {}
    for c_id, idx_list in cluster_indices.items():
        crops = [cv2.resize(glyphs[i]['image'].astype(np.float32), target_size) for i in idx_list]
        avg_img = np.mean(crops, axis=0).astype(np.uint8)
        reps[c_id] = avg_img
    return reps
