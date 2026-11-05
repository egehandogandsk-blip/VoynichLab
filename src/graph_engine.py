"""
Voynich Graph Layout Engine
---------------------------
Scale-adaptive page analysis built on a component graph:

  ink components --(spatial graph)--> line segments --(gap clustering)--> words
  line segments --(vertical proximity)--> blocks (paragraphs)
  line segments --(same baseline across figures)--> logical lines (reading order)

Everything is expressed relative to the page's own glyph scale `m` (median component
height), so the same code works for a crop, a full page or a high-resolution PDF render.
Words can then be aligned to transcription tokens (width-aware dynamic programming) and
pages can be matched against the whole manuscript by their line-length profile.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.eva import unit_len

DEFAULT_PARAMS = {
    'sensitivity': 1.15,       # tuned for medieval ink contrast
    'min_line_components': 2,
    'min_text_score': 0.24,    # allow interrupted short lines
    'word_gap_scale': 1.0,     # multiplies the automatically estimated word-gap threshold
    'max_component_height': 3.8,  # allow tall gallows letters (t, p, f, k)
    'paint_reject': 0.45,
}


# ----------------------------------------------------------------------------- utils
class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _otsu_1d(values: np.ndarray, bins: int = 48) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(np.median(values))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    mt = (hist * centers).sum()
    m1 = (mt - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def _union_bbox(boxes: np.ndarray) -> Tuple[int, int, int, int]:
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


# ----------------------------------------------------------------------- ink / paint
def extract_ink_and_paint(img_bgr: np.ndarray, sensitivity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ink_mask, paint_mask) as uint8 {0,255}."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = max(9, int(round(min(gray.shape) / 20)) | 1)

    # Background illumination estimation via morphological opening + blur
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255)

    # Gradient edge magnitude to distinguish sharp ink strokes from soft watercolor wash
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive multi-scale threshold
    win = 29
    mean = cv2.boxFilter(norm, -1, (win, win))
    sq = cv2.boxFilter(norm * norm, -1, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    kk = float(np.clip(0.25 - 0.14 * (sensitivity - 1.0), 0.08, 0.45))
    thresh = mean * (1 + kk * (std / 128.0 - 1))

    # Ink: either strong local dark threshold OR significant edge gradient in non-white area
    ink_candidates = ((norm < thresh) & (norm < 228)) | ((norm < mean * 0.95) & (grad_norm > 16))

    # Color analysis in HSV: watercolor paints (green, cyan, blue, red/brown washes)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    s_med = int(np.median(s))

    # Green/blue/teal foliage and stems (watercolor paint washes are luminous, v > 65)
    cool = (h >= 28) & (h <= 145) & (s > s_med + 18) & (v > 65)
    # Red/orange floral pigment
    red = ((h <= 12) | (h >= 160)) & (s > s_med + 45) & (v > 65)
    paint = ((cool | red).astype(np.uint8)) * 255
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    paint = cv2.dilate(paint, np.ones((4, 4), np.uint8))

    # Ink: must not be watercolor wash, or must be very dark stroke (v < 55)
    is_dark_ink = ink_candidates & (v < 55)
    is_light_ink = ink_candidates & (s < 75) & (paint == 0)
    ink = ((is_dark_ink | is_light_ink).astype(np.uint8)) * 255
    # Light cleanup of isolated single-pixel noise
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    return ink, paint


# ----------------------------------------------------------------------- page canvas & boundary
def detect_page_canvas(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detects the physical parchment/paper boundary against dark scanner beds, shadows, or scanner covers.
    Returns (clean_img, page_bbox).
    Only dark regions connected to the image outer borders with area > 0.005 * H * W are treated as scanner cradle.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    is_dark = (gray < 45) & (img_bgr[:, :, 0] < 50) & (img_bgr[:, :, 1] < 50) & (img_bgr[:, :, 2] < 50)
    if np.count_nonzero(is_dark) < 0.005 * H * W:
        return img_bgr.copy(), (0, 0, W, H)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(is_dark.astype(np.uint8) * 255)
    bed_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, n_lab):
        x, y, w, h, a = stats[i]
        touches_border = (x == 0 or y == 0 or x + w >= W or y + h >= H)
        if touches_border and a > 0.005 * H * W:
            bed_mask[labels == i] = 255

    if np.count_nonzero(bed_mask) == 0:
        return img_bgr.copy(), (0, 0, W, H)

    bed_mask = cv2.morphologyEx(bed_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bed_mask = cv2.dilate(bed_mask, np.ones((11, 11), np.uint8))

    parch_mask = (bed_mask == 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(parch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        page_bbox = cv2.boundingRect(c)
    else:
        page_bbox = (0, 0, W, H)

    parch_pixels = img_bgr[bed_mask == 0]
    parch_color = np.median(parch_pixels, axis=0).astype(np.uint8) if len(parch_pixels) > 0 else np.array([210, 210, 210], dtype=np.uint8)

    clean_img = img_bgr.copy()
    clean_img[bed_mask > 0] = parch_color
    return clean_img, page_bbox


def extract_illustrations(paint: np.ndarray, m: float) -> List[Dict[str, Any]]:
    """Detects contiguous botanical drawings, large figures, and watercolor pigment clusters.
    Unifies leaves, stems, flower heads, and roots into coherent botanical regions.
    """
    if paint is None or np.count_nonzero(paint) == 0:
        return []
    H, W = paint.shape[:2]

    # Morphological closing with ellipse kernel scaled to glyph size to bridge leaves and stems
    close_k = max(15, int(round(3.5 * m)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    paint_closed = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, kernel)

    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(paint_closed, connectivity=8)
    raw_boxes = []
    min_area = max(500, int(35 * m * m))
    for i in range(1, n_lab):
        area = stats[i, cv2.CC_STAT_AREA]
        pw = stats[i, cv2.CC_STAT_WIDTH]
        ph = stats[i, cv2.CC_STAT_HEIGHT]
        px = stats[i, cv2.CC_STAT_LEFT]
        py = stats[i, cv2.CC_STAT_TOP]
        if area > min_area or (pw > 5.0 * m and ph > 5.0 * m and area > 15 * m * m):
            raw_boxes.append([px, py, px + pw, py + ph])

    if not raw_boxes:
        return []

    # Merge overlapping or touching illustration bounding boxes
    merged: List[List[int]] = []
    for b in raw_boxes:
        x0, y0, x1, y1 = b
        pad = int(1.5 * m)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)

        merged_with = -1
        for mi, mb in enumerate(merged):
            if not (x1 < mb[0] or x0 > mb[2] or y1 < mb[1] or y0 > mb[3]):
                merged_with = mi
                break
        if merged_with >= 0:
            mb = merged[merged_with]
            merged[merged_with] = [min(mb[0], x0), min(mb[1], y0), max(mb[2], x1), max(mb[3], y1)]
        else:
            merged.append([x0, y0, x1, y1])

    ills = []
    for mb in merged:
        bx0, by0, bx1, by1 = mb
        bw, bh = bx1 - bx0, by1 - by0
        ills.append({
            'bbox': [int(bx0), int(by0), int(bw), int(bh)],
            'area': int(bw * bh),
            'type': 'botanical_illustration'
        })
    return ills


def _estimate_scale(stats: np.ndarray) -> float:
    if stats.shape[0] <= 1:
        return 14.0
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    a = stats[1:, cv2.CC_STAT_AREA]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    sel = h[(h >= 8) & (h <= 75) & (a >= 20) & (w >= 3)]
    if sel.size < 6:
        sel = h[(h >= 5) & (h <= 80)]
        if sel.size < 4:
            return 14.0
    return float(np.median(sel))


# ---------------------------------------------------------------------------- main
def analyze_layout(img_bgr: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    H, W = img_bgr.shape[:2]

    # 1. Page canvas detection & dark scanner bed neutralization
    clean_img, page_bbox = detect_page_canvas(img_bgr)
    px0, py0, pw0, ph0 = page_bbox

    # 2. Extract ink and paint from the sanitized page canvas
    ink, paint = extract_ink_and_paint(clean_img, p['sensitivity'])

    # Enforce boundary: Zero out ink/paint outside parchment contour
    ink_mask_page = np.zeros((H, W), dtype=np.uint8)
    ink_mask_page[py0:py0 + ph0, px0:px0 + pw0] = 255
    ink = cv2.bitwise_and(ink, ink_mask_page)
    paint = cv2.bitwise_and(paint, ink_mask_page)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    m = _estimate_scale(stats)

    # 3. Detect unified illustrations
    illustrations = extract_illustrations(paint, m)

    def in_illustration(cx_val: float, cy_val: float) -> bool:
        for ill in illustrations:
            bx, by, bw, bh = ill['bbox']
            if bx <= cx_val <= bx + bw and by <= cy_val <= by + bh:
                return True
        return False

    # 4. Component filtering: drop sub-pixel dust, margin artifacts, and components inside illustrations
    margin_x = max(int(0.025 * W), int(1.5 * m))
    margin_y = max(int(0.025 * H), int(1.5 * m))

    x, y = stats[:, 0], stats[:, 1]
    w, h = stats[:, 2], stats[:, 3]
    area = stats[:, 4]
    integ = cv2.integral((paint > 0).astype(np.uint8))

    def paint_frac(i: int) -> float:
        x0, y0, x1, y1 = x[i], y[i], x[i] + w[i], y[i] + h[i]
        s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        return float(s) / max(1, w[i] * h[i])

    keep_idx = []
    rejected_big = rejected_paint = 0
    for i in range(1, n_lab):
        if area[i] < max(8, 0.05 * m * m) or h[i] < 5 or w[i] < 2:
            continue
        if h[i] > p['max_component_height'] * m or w[i] > 9 * m:
            rejected_big += 1
            continue
        if paint_frac(i) > p['paint_reject']:
            rejected_paint += 1
            continue
        comp_cx = x[i] + w[i] / 2.0
        comp_cy = y[i] + h[i] / 2.0
        if in_illustration(comp_cx, comp_cy):
            continue
        # Discard outer margin noise
        if comp_cx < px0 + margin_x or comp_cx > px0 + pw0 - margin_x:
            continue
        if comp_cy < py0 + margin_y or comp_cy > py0 + ph0 - margin_y:
            continue
        keep_idx.append(i)

    comps = np.array([[x[i], y[i], x[i] + w[i], y[i] + h[i]] for i in keep_idx], dtype=np.int32) \
        if keep_idx else np.zeros((0, 4), dtype=np.int32)
    n = len(comps)

    result: Dict[str, Any] = {
        'image_size': [W, H], 'm': m, 'params': p,
        'components': comps.tolist(), 'words': [], 'segments': [], 'blocks': [],
        'logical_lines': [], 'edges': [], 'illustrations': illustrations,
        'stats': {'components': n, 'rejected_large': rejected_big, 'rejected_paint': rejected_paint,
                  'paint_coverage': float((paint > 0).mean()), 'ink_coverage': float((ink > 0).mean()),
                  'illustrations_detected': len(illustrations)},
    }
    if n < 3:
        return result

    # 5. Component graph (numpy, chunked DSU with tight vertical overlap)
    cx = (comps[:, 0] + comps[:, 2]) / 2.0
    cy = (comps[:, 1] + comps[:, 3]) / 2.0
    ch = (comps[:, 3] - comps[:, 1]).astype(float)
    dsu = _DSU(n)
    max_gap = 2.8 * m
    edges: List[Tuple[int, int]] = []
    order = np.argsort(comps[:, 0])
    cs = comps[order]
    for a in range(n):
        ia = order[a]
        hi = np.searchsorted(cs[:, 0], comps[ia, 2] + max_gap, side='right')
        if hi <= a + 1:
            continue
        cand = order[a + 1:hi]
        gap = comps[cand, 0] - comps[ia, 2]

        top_overlap = np.maximum(0, np.minimum(comps[cand, 3], comps[ia, 3]) - np.maximum(comps[cand, 1], comps[ia, 1]))
        min_h = np.minimum(ch[cand], ch[ia])
        v_overlap = (top_overlap >= 0.15 * min_h)

        dy = np.abs(cy[cand] - cy[ia])
        max_allowed_dy = 0.45 * np.maximum(m, min_h)

        ok = v_overlap & (dy <= max_allowed_dy) & (gap >= -0.3 * m) & (gap <= max_gap)
        for j in cand[ok]:
            dsu.union(ia, int(j))
            edges.append((ia, int(j)))

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)

    # 6. Line segments: filter out isolated margin dust (require length and components)
    segs: List[Dict[str, Any]] = []
    assigned_comps: set = set()
    for ids in groups.values():
        b = comps[ids]
        bx, by, bw, bh = _union_bbox(b)
        heights = ch[ids]
        in_band = float(np.mean((heights >= 0.35 * m) & (heights <= 2.4 * m)))
        aspect = min(1.0, (bw / max(bh, 1)) / 2.5)
        size_term = min(1.0, len(ids) / 5.0)
        score = 0.45 * in_band + 0.3 * aspect + 0.25 * size_term
        if bw < 3.2 * m or len(ids) < p['min_line_components'] or score < p['min_text_score']:
            continue
        segs.append({'ids': set(ids), 'bbox': [bx, by, bw, bh],
                     'score': score, 'cy': float(cy[ids].mean())})
        assigned_comps.update(ids)

    if not segs:
        return result

    # 7. Ascender / Descender Reassociation:
    # Any component not yet assigned that sits vertically directly above or below a line
    # within the line's horizontal span is attached to that line!
    for i in range(n):
        if i in assigned_comps:
            continue
        ci_cx, ci_cy = cx[i], cy[i]
        best_seg = None
        min_dist = 1.2 * m
        for s in segs:
            sbx, sby, sbw, sbh = s['bbox']
            if sbx - 0.5 * m <= ci_cx <= sbx + sbw + 0.5 * m:
                dy = abs(ci_cy - s['cy'])
                if dy < min_dist:
                    min_dist = dy
                    best_seg = s
        if best_seg is not None:
            best_seg['ids'].add(i)
            assigned_comps.add(i)

    # Recompute line bboxes
    final_segs: List[Dict[str, Any]] = []
    for s in segs:
        cids = sorted(list(s['ids']), key=lambda k: comps[k, 0])
        b = comps[cids]
        bx, by, bw, bh = _union_bbox(b)
        final_segs.append({'ids': cids, 'bbox': [bx, by, bw, bh], 'score': s['score'], 'cy': float(cy[cids].mean())})
    final_segs.sort(key=lambda s: (s['cy'], s['bbox'][0]))

    # 8. Collinear line segment merging: Merge segments on the same baseline with horizontal gap <= 4.8 * m
    ldsu = _DSU(len(final_segs))
    for i in range(len(final_segs)):
        bi = final_segs[i]['bbox']
        for j in range(i + 1, len(final_segs)):
            bj = final_segs[j]['bbox']
            dy = abs(final_segs[j]['cy'] - final_segs[i]['cy'])
            if dy <= 0.45 * m:
                gap = max(bj[0] - (bi[0] + bi[2]), bi[0] - (bj[0] + bj[2]))
                if gap <= 4.8 * m:
                    ldsu.union(i, j)

    merged_lines_map: Dict[int, List[int]] = {}
    for i in range(len(final_segs)):
        merged_lines_map.setdefault(ldsu.find(i), []).append(i)

    segs = []
    for seg_indices in merged_lines_map.values():
        all_cids = sorted([cid for si in seg_indices for cid in final_segs[si]['ids']], key=lambda k: comps[k, 0])
        b = comps[all_cids]
        bx, by, bw, bh = _union_bbox(b)
        segs.append({'ids': all_cids, 'bbox': [bx, by, bw, bh], 'cy': float(cy[all_cids].mean()), 'score': 1.0})
    segs.sort(key=lambda s: s['cy'])

    # 9. Word gap threshold (pooled gaps, Otsu)
    pooled: List[float] = []
    for s in segs:
        runmax = None
        for k in s['ids']:
            if runmax is not None:
                g = comps[k, 0] - runmax
                if g > 0:
                    pooled.append(float(g))
            runmax = comps[k, 2] if runmax is None else max(runmax, comps[k, 2])
    pooled_arr = np.clip(np.array(pooled), 0, 2.5 * m) if pooled else np.array([])
    if pooled_arr.size >= 15:
        thr = _otsu_1d(pooled_arr)
    else:
        thr = 0.42 * m
    thr = float(np.clip(thr, 0.28 * m, 0.75 * m)) * p['word_gap_scale']
    result['stats']['word_gap_threshold'] = thr

    # 10. Individual word segmentation with internal split safeguard
    words: List[Dict[str, Any]] = []
    for si, s in enumerate(segs):
        cur: List[int] = []
        runmax = None
        gaps_in: List[float] = []
        pending: List[Tuple[List[int], List[float], float]] = []
        for k in s['ids']:
            if runmax is not None and comps[k, 0] - runmax > thr:
                pending.append((cur, gaps_in, comps[k, 0] - runmax))
                cur, gaps_in = [], []
            elif runmax is not None:
                gaps_in.append(max(0.0, float(comps[k, 0] - runmax)))
            cur.append(k)
            runmax = comps[k, 2] if runmax is None else max(runmax, comps[k, 2])
        if cur:
            pending.append((cur, gaps_in, float('inf')))

        prev_gap = float('inf')
        for cids, gin, next_gap in pending:
            if not cids:
                continue
            bb = _union_bbox(comps[cids])

            # Safeguard against multi-word agglomerations
            if bb[2] > 4.8 * m and len(cids) > 1:
                sub_cur = [cids[0]]
                sub_runmax = comps[cids[0], 2]
                sub_words_cids = []
                for cid in cids[1:]:
                    cg = comps[cid, 0] - sub_runmax
                    if cg > 0.25 * m:
                        sub_words_cids.append(sub_cur)
                        sub_cur = [cid]
                    else:
                        sub_cur.append(cid)
                    sub_runmax = max(sub_runmax, comps[cid, 2])
                if sub_cur:
                    sub_words_cids.append(sub_cur)

                for sc in sub_words_cids:
                    sbb = _union_bbox(comps[sc])
                    words.append({'bbox': list(sbb), 'seg': si, 'comps': sc, 'conf': 0.85})
                continue

            bound = min(prev_gap, next_gap)
            bound_conf = 0.5 if bound == float('inf') else float(np.clip(bound / thr, 1, 3) / 3 + 0.17)
            inner = max(gin) if gin else 0.0
            inner_conf = 1.0 - float(np.clip(inner / max(thr, 1e-6), 0, 1))
            conf = float(np.clip(0.5 * min(1.0, bound_conf) + 0.5 * inner_conf, 0, 1))
            words.append({'bbox': list(bb), 'seg': si, 'comps': cids, 'conf': conf})
            prev_gap = next_gap

    # 11. Paragraph Block Clustering (Group vertically stacked lines into coherent blocks)
    sdsu = _DSU(len(segs))
    for i in range(len(segs)):
        bi = segs[i]['bbox']
        for j in range(i + 1, len(segs)):
            bj = segs[j]['bbox']
            if segs[j]['cy'] - segs[i]['cy'] > 3.5 * m:
                break
            ox = min(bi[0] + bi[2], bj[0] + bj[2]) - max(bi[0], bj[0])
            if ox <= 0.20 * min(bi[2], bj[2]):
                continue
            sdsu.union(i, j)

    bgroups: Dict[int, List[int]] = {}
    for i in range(len(segs)):
        bgroups.setdefault(sdsu.find(i), []).append(i)
    blocks = []
    for bi_, ids in enumerate(sorted(bgroups.values(), key=lambda g: min(segs[k]['bbox'][1] for k in g))):
        bb = np.array([[segs[k]['bbox'][0], segs[k]['bbox'][1],
                        segs[k]['bbox'][0] + segs[k]['bbox'][2], segs[k]['bbox'][1] + segs[k]['bbox'][3]] for k in ids])
        blocks.append({'lines': ids, 'bbox': list(_union_bbox(bb))})
        for k in ids:
            segs[k]['block'] = bi_

    # 12. Natural Reading Order: Blocks top-to-bottom, lines top-to-bottom, words left-to-right
    word_ids_by_seg: Dict[int, List[int]] = {}
    for wi, wd in enumerate(words):
        word_ids_by_seg.setdefault(wd['seg'], []).append(wi)

    ll_out = []
    order_counter = 0
    for bi_, blk in enumerate(blocks):
        blk_seg_ids = sorted(blk['lines'], key=lambda si: segs[si]['cy'])
        for li, si in enumerate(blk_seg_ids):
            wids = word_ids_by_seg.get(si, [])
            wids.sort(key=lambda wi: words[wi]['bbox'][0])
            for wi in wids:
                words[wi]['block'] = bi_
                words[wi]['line_in_block'] = li
                words[wi]['order'] = order_counter
                order_counter += 1
            if wids:
                w_boxes = np.array([[*words[wi]['bbox'][:2], words[wi]['bbox'][0] + words[wi]['bbox'][2], words[wi]['bbox'][1] + words[wi]['bbox'][3]] for wi in wids])
                ll_out.append({'segments': [si], 'word_ids': wids, 'bbox': list(_union_bbox(w_boxes)), 'block': bi_})

    for si, s in enumerate(segs):
        s['words'] = word_ids_by_seg.get(si, [])

    result.update({
        'words': words,
        'segments': [{k: v for k, v in s.items() if k != 'ids'} | {'n_components': len(s['ids'])} for s in segs],
        'blocks': blocks,
        'logical_lines': ll_out,
        'edges': [[int(a), int(b)] for a, b in edges[:20000]],
    })
    word_conf = [wd['conf'] for wd in words]
    result['stats'].update({
        'segments': len(segs), 'blocks': len(blocks), 'logical_lines': len(ll_out), 'words': len(words),
        'mean_word_conf': float(np.mean(word_conf)) if word_conf else 0.0,
        'words_per_line': [len(l['word_ids']) for l in ll_out],
    })
    return result


layout_page = analyze_layout



# ---------------------------------------------------------------------- overlays
def draw_overlay(img_bgr: np.ndarray, layout: Dict[str, Any], layers: Optional[Dict[str, bool]] = None,
                 highlight_word: Optional[int] = None, labels: Optional[Dict[int, str]] = None) -> np.ndarray:
    layers = layers or {'words': True, 'lines': True, 'blocks': True, 'graph': False, 'components': False, 'order': False}
    out = img_bgr.copy()
    comps = layout.get('components', [])
    if layers.get('components'):
        for x0, y0, x1, y1 in comps:
            cv2.rectangle(out, (x0, y0), (x1, y1), (180, 180, 180), 1)
    if layers.get('graph'):
        for a, b in layout.get('edges', []):
            ca, cb = comps[a], comps[b]
            cv2.line(out, ((ca[0] + ca[2]) // 2, (ca[1] + ca[3]) // 2),
                     ((cb[0] + cb[2]) // 2, (cb[1] + cb[3]) // 2), (255, 0, 255), 1)
    if layers.get('blocks'):
        for b in layout.get('blocks', []):
            x, y, w, h = b['bbox']
            cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 140, 255), 2)
    if layers.get('lines'):
        for s in layout.get('segments', []):
            x, y, w, h = s['bbox']
            cv2.rectangle(out, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 120, 0), 1)
    if layers.get('words'):
        for wi, wd in enumerate(layout.get('words', [])):
            x, y, w, h = wd['bbox']
            c = wd['conf']
            color = (0, int(120 + 110 * c), int(220 * (1 - c)))   # BGR: low conf -> yellowish, high -> green
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 1 if wi != highlight_word else 3)
            if layers.get('order'):
                cv2.putText(out, str(wd.get('order', wi)), (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            if labels and wi in labels:
                cv2.putText(out, labels[wi], (x, y + h + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 0, 0), 1)
    if layers.get('illustrations', True):
        for ill in layout.get('illustrations', []):
            ix, iy, iw, ih = ill['bbox']
            cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), (255, 0, 200), 2)
            cv2.putText(out, "ILLUSTRASYON", (ix + 4, max(12, iy + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 200), 1)
    if highlight_word is not None and 0 <= highlight_word < len(layout.get('words', [])):
        x, y, w, h = layout['words'][highlight_word]['bbox']
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
    return out


def word_crop(img_bgr: np.ndarray, word: Dict[str, Any], pad: int = 4) -> np.ndarray:
    x, y, w, h = word['bbox']
    H, W = img_bgr.shape[:2]
    return img_bgr[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]


# -------------------------------------------------------------- alignment & matching
def _line_align(det: List[int], ref: List[int], skip_det: float = 1.0, skip_ref: float = 0.6,
                free_ref_ends: bool = False) -> Tuple[float, List[Tuple[int, int]]]:
    """DP alignment of two sequences of per-line word counts. Returns (cost, [(det_i, ref_j)])."""
    n, m_ = len(det), len(ref)
    INF = 1e18
    D = np.full((n + 1, m_ + 1), INF)
    B = np.zeros((n + 1, m_ + 1), dtype=np.int8)
    D[0, 0] = 0.0
    for j in range(1, m_ + 1):
        D[0, j] = 0.0 if free_ref_ends else D[0, j - 1] + skip_ref
        B[0, j] = 2
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + skip_det
        B[i, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m_ + 1):
            a, b = det[i - 1], ref[j - 1]
            c_match = D[i - 1, j - 1] + abs(a - b) / max(a, b, 1)
            c_sd = D[i - 1, j] + skip_det
            c_sr = D[i, j - 1] + (0.0 if free_ref_ends and i == n else skip_ref)
            best = min(c_match, c_sd, c_sr)
            D[i, j] = best
            B[i, j] = 0 if best == c_match else (1 if best == c_sd else 2)
    if free_ref_ends:
        j_end = int(np.argmin(D[n, :]))
        total = float(D[n, j_end])
    else:
        j_end, total = m_, float(D[n, m_])
    i, j = n, j_end
    pairs: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and B[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or B[i, j] == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return total, pairs


def _word_token_align(widths: List[float], tokens: List[str], scale: float) -> List[Tuple[List[int], List[int], float]]:
    """Monotone alignment of detected words (by width) to tokens (by glyph-unit length).
    Returns list of (word_idx_list, token_idx_list, confidence)."""
    n, m_ = len(widths), len(tokens)
    exp = [unit_len(t) * scale for t in tokens]
    INF = 1e18
    D = np.full((n + 1, m_ + 1), INF)
    B: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    D[0, 0] = 0
    for i in range(n + 1):
        for j in range(m_ + 1):
            cur = D[i, j]
            if cur >= INF:
                continue
            if i < n and j < m_:      # 1-1
                c = cur + abs(widths[i] - exp[j]) / max(widths[i], exp[j], 1e-6)
                if c < D[i + 1, j + 1]:
                    D[i + 1, j + 1] = c
                    B[(i + 1, j + 1)] = (i, j, 1, 1)
            if i + 1 < n and j < m_:  # 2 words -> 1 token
                wsum = widths[i] + widths[i + 1]
                c = cur + abs(wsum - exp[j]) / max(wsum, exp[j], 1e-6) + 0.2
                if c < D[i + 2, j + 1]:
                    D[i + 2, j + 1] = c
                    B[(i + 2, j + 1)] = (i, j, 2, 1)
            if i < n and j + 1 < m_:  # 1 word -> 2 tokens
                esum = exp[j] + exp[j + 1]
                c = cur + abs(widths[i] - esum) / max(widths[i], esum, 1e-6) + 0.2
                if c < D[i + 1, j + 2]:
                    D[i + 1, j + 2] = c
                    B[(i + 1, j + 2)] = (i, j, 1, 2)
            if i < n:                 # spurious word
                c = cur + 1.0
                if c < D[i + 1, j]:
                    D[i + 1, j] = c
                    B[(i + 1, j)] = (i, j, 1, 0)
            if j < m_:                # undetected token
                c = cur + 1.0
                if c < D[i, j + 1]:
                    D[i, j + 1] = c
                    B[(i, j + 1)] = (i, j, 0, 1)
    out: List[Tuple[List[int], List[int], float]] = []
    i, j = n, m_
    while (i, j) != (0, 0):
        pi, pj, dw, dt = B[(i, j)]
        step_cost = D[i, j] - D[pi, pj]
        out.append((list(range(pi, i)), list(range(pj, j)), float(max(0.0, 1.0 - step_cost))))
        i, j = pi, pj
    out.reverse()
    return out


def align_words_to_tokens(layout: Dict[str, Any], zl_lines: List[List[str]]) -> Dict[str, Any]:
    """Two-level alignment: logical lines <-> transcription lines, then words <-> tokens."""
    lls = layout.get('logical_lines', [])
    words = layout.get('words', [])
    if not lls or not zl_lines:
        return {'word_tokens': {}, 'line_pairs': [], 'quality': 0.0, 'unmatched_tokens': 0}
    det_counts = [len(l['word_ids']) for l in lls]
    ref_counts = [len(l) for l in zl_lines]
    # crops show only part of a page -> allow free ref ends when the detection is much smaller
    free = sum(det_counts) < 0.8 * sum(ref_counts)
    cost, pairs = _line_align(det_counts, ref_counts, free_ref_ends=free)

    # global scale: pixels per glyph unit
    tot_w = sum(words[wi]['bbox'][2] for di, _ in pairs for wi in lls[di]['word_ids'])
    tot_u = sum(unit_len(t) for _, rj in pairs for t in zl_lines[rj])
    scale = tot_w / max(tot_u, 1)

    mapping: Dict[int, Dict[str, Any]] = {}
    confs: List[float] = []
    matched_tokens = 0
    for di, rj in pairs:
        wids = lls[di]['word_ids']
        toks = zl_lines[rj]
        widths = [float(words[wi]['bbox'][2]) for wi in wids]
        for widx_list, tidx_list, c in _word_token_align(widths, toks, scale):
            if not widx_list or not tidx_list:
                continue
            matched_tokens += len(tidx_list)
            for wi_local in widx_list:
                wi = wids[wi_local]
                mapping[wi] = {'tokens': [toks[t] for t in tidx_list], 'conf': c, 'zl_line': rj, 'det_line': di}
                confs.append(c)
    total_tokens = sum(ref_counts[rj] for _, rj in pairs)
    return {
        'word_tokens': mapping,
        'line_pairs': pairs,
        'line_cost': cost,
        'quality': float(np.mean(confs)) if confs else 0.0,
        'coverage': matched_tokens / max(total_tokens, 1),
        'unmatched_tokens': total_tokens - matched_tokens,
        'scale_px_per_unit': scale,
        'free_ends': free,
    }


def find_matching_folios(layout: Dict[str, Any], zl: Dict[str, Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    """Rank manuscript folios by how well their per-line token counts match the detected word counts."""
    det = layout.get('stats', {}).get('words_per_line') or []
    if len(det) < 2:
        return []
    results = []
    for fid, rec in zl.items():
        ref = [len(l['tokens']) for l in rec['lines'] if l['tokens']]
        if len(ref) < 2:
            continue
        cost, pairs = _line_align(det, ref, skip_det=1.0, skip_ref=0.6, free_ref_ends=True)
        norm = cost / len(det)
        results.append({'folio': fid, 'score': float(1.0 / (1.0 + norm)), 'cost': float(cost),
                        'matched_lines': len(pairs), 'ref_lines': len(ref),
                        'section': rec['section'], 'language': rec['language']})
    results.sort(key=lambda r: -r['score'])
    return results[:top_k]
