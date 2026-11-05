"""
Voynich PDF Corpus Builder
Renders an uploaded PDF page by page, assigns each page a folio id, analyzes every page
(vision + transcription-based statistics + glyph signatures) and stores everything in a
persistent corpus (data/corpus/<slug>/corpus.json).
"""

import os
import re
import json
import shutil
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from src.vision_segmenter import load_image, binarize_page, process_page
from src.glyph_clustering import extract_glyphs_from_word, extract_features
from src.pipeline import load_folio_transcription
from src.slot_grammar import analyze_corpus_morphology
from src.crypto_entropy import (
    compute_shannon_entropy, compute_index_of_coincidence,
    analyze_zipf_law, classify_currier_dialect,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_ROOT = os.path.join(BASE_DIR, 'data', 'corpus')
DEFAULT_TRANSCRIPT = os.path.join(BASE_DIR, 'data', 'transcripts', 'ZL3b-n.txt')

# Folios that do not exist in MS 408 (lost / removed)
MISSING_FOLIOS = {12, 59, 60, 61, 62, 63, 64, 74, 91, 92, 97, 98, 109, 110}
LAST_FOLIO = 116

SECTIONS = [
    (1, 66, 'Bitkiler (Herbal)'),
    (67, 73, 'Astronomi / Astroloji'),
    (75, 84, 'Biyolojik (Banyo)'),
    (85, 86, 'Kozmolojik (Rozet)'),
    (87, 102, 'Eczacılık (Pharmaceutical)'),
    (103, 116, 'Reçeteler (Stars / Recipes)'),
]


def section_of(folio_number: int) -> str:
    for lo, hi, name in SECTIONS:
        if lo <= folio_number <= hi:
            return name
    return 'Bilinmiyor'


def generate_folio_sequence(count: int, start_folio: int = 1, start_side: str = 'r') -> List[str]:
    """Sequential folio labels (f1r, f1v, f2r ...) skipping folios missing from the manuscript."""
    labels: List[str] = []
    num, side = start_folio, start_side
    while len(labels) < count and num <= LAST_FOLIO:
        if num in MISSING_FOLIOS:
            num += 1
            side = 'r'
            continue
        labels.append(f'f{num}{side}')
        if side == 'r':
            side = 'v'
        else:
            side = 'r'
            num += 1
    # If PDF has more pages than the manuscript, mark the rest as extra pages
    while len(labels) < count:
        labels.append(f'extra{len(labels) + 1}')
    return labels


def slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r'[^A-Za-z0-9_-]+', '_', base).strip('_') or 'pdf_corpus'


def _notify(cb: Optional[Callable], p: float, msg: str, img_path: Optional[str] = None) -> None:
    if not cb:
        return
    try:
        import inspect
        sig = inspect.signature(cb)
        if len(sig.parameters) >= 3:
            cb(p, msg, img_path)
        else:
            cb(p, msg)
    except Exception:
        pass


def render_pdf_pages(pdf_path: str, out_dir: str, target_width: int = 1200,
                     progress_cb: Optional[Callable] = None) -> List[str]:
    """Render each PDF page to PNG. Returns list of image paths in page order."""
    import fitz  # PyMuPDF

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths: List[str] = []
    total = len(doc)
    for i in range(total):
        page = doc[i]
        zoom = target_width / max(1.0, page.rect.width)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        path = os.path.join(out_dir, f'p{i + 1:04d}.png')
        pix.save(path)
        paths.append(path)
        _notify(progress_cb, (i + 1) / total * 0.4, f'Sayfa {i + 1}/{total} görsele çevriliyor', path)
    doc.close()
    return paths


def read_page_header(transcript_path: str) -> Dict[str, Dict[str, str]]:
    """Parse ZL header lines: language, hand, plant id, for every folio."""
    info: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(transcript_path):
        return info
    with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    for m in re.finditer(r'<f(\d+[rv]\d*)>\s*<!([^>]*)>([\s\S]*?)(?=<f\d+[rv]\d*\.)', text):
        fid, tags, comments = m.group(1), m.group(2), m.group(3)
        lang = re.search(r'\$L=(\w)', tags)
        hand = re.search(r'\$H=(\w)', tags)
        plant = re.search(r'# Plant ID:\s*([^\n]+)', comments)
        info[fid] = {
            'language': lang.group(1) if lang else '',
            'hand': hand.group(1) if hand else '',
            'plant_id': plant.group(1).strip() if plant else '',
        }
    return info


def analyze_page(image_path: str, folio_id: str, header_info: Dict[str, Dict[str, str]],
                 transcript_path: str, max_glyphs: int = 120) -> Dict[str, Any]:
    """Analyze one page. Returns the corpus record plus glyph feature matrix (not JSON-stored)."""
    clean = folio_id.lower().replace('f', '', 1)
    m = re.match(r'(\d+)', clean)
    number = int(m.group(1)) if m else 0

    entry: Dict[str, Any] = {
        'folio_id': folio_id,
        'folio_number': number,
        'image_path': image_path,
        'section': section_of(number) if m else 'Ek Sayfa',
        'vision': {},
        'transcription': {'available': False},
        'header': header_info.get(clean, {}),
    }

    # Vision layer
    glyph_features = np.zeros((0, 0))
    try:
        seg = process_page(image_path)
        entry['vision'] = {
            'lines': seg['line_count'],
            'words': seg['word_count'],
            'strokes': seg['stroke_count'],
            'image_size': [int(seg['image_shape'][1]), int(seg['image_shape'][0])],
        }
        img = load_image(image_path)
        bin_img = binarize_page(img)
        feats = []
        words = seg['words']
        step = max(1, len(words) // 40)
        for w in words[::step]:
            for g in extract_glyphs_from_word(bin_img, w['bbox']):
                feats.append(extract_features(g['image']))
                if len(feats) >= max_glyphs:
                    break
            if len(feats) >= max_glyphs:
                break
        if feats:
            glyph_features = np.nan_to_num(np.array(feats))
    except Exception as exc:  # keep batch going
        entry['vision'] = {'error': str(exc)}

    # Transcription-based layer (only for folios present in ZL)
    lines = load_folio_transcription(transcript_path, clean) if m else []
    tokens = [t for l in lines for t in l['tokens']]
    if tokens:
        text = ' '.join(tokens)
        ent = compute_shannon_entropy(text)
        zipf = analyze_zipf_law(tokens)
        morph = analyze_corpus_morphology(tokens)
        curr = classify_currier_dialect(tokens)
        entry['transcription'] = {
            'available': True,
            'lines': len(lines),
            'tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'h1': ent['h1'],
            'h2': ent['h2'],
            'ic': compute_index_of_coincidence(text),
            'zipf_alpha': zipf['alpha'],
            'slot_valid_rate': morph['valid_rate'],
            'currier_estimate': curr['dialect'],
            'first_line': lines[0]['clean_text'] if lines else '',
            'token_list': tokens,
        }
    return {'entry': entry, 'glyph_features': glyph_features}


def cluster_corpus_glyphs(feature_blocks: List[np.ndarray], n_clusters: int = 24) -> Dict[str, Any]:
    """Cluster glyph features of the whole corpus; return per-page cluster histograms."""
    from sklearn.cluster import MiniBatchKMeans

    valid = [(i, b) for i, b in enumerate(feature_blocks) if b.size]
    if not valid:
        return {'n_clusters': 0, 'histograms': {}}
    stacked = np.vstack([b for _, b in valid])
    k = max(2, min(n_clusters, len(stacked)))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024).fit(stacked)

    hist: Dict[int, List[float]] = {}
    for i, b in valid:
        labels = km.predict(b)
        h = np.bincount(labels, minlength=k).astype(float)
        hist[i] = (h / h.sum()).round(4).tolist()
    return {'n_clusters': k, 'total_glyphs': int(len(stacked)), 'histograms': hist}


def build_corpus(pdf_path: str, corpus_name: Optional[str] = None, start_folio: int = 1,
                 start_side: str = 'r', page_offset: int = 0,
                 transcript_path: str = DEFAULT_TRANSCRIPT,
                 progress_cb: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """
    Full PDF -> corpus pipeline.
    page_offset: number of leading PDF pages to skip (covers, binding pages).
    """
    slug = slugify(corpus_name or pdf_path)
    corpus_dir = os.path.join(CORPUS_ROOT, slug)
    pages_dir = os.path.join(corpus_dir, 'pages')
    if os.path.isdir(pages_dir):
        shutil.rmtree(pages_dir)

    all_paths = render_pdf_pages(pdf_path, pages_dir, progress_cb=progress_cb)
    cover_paths = all_paths[:page_offset]
    page_paths = all_paths[page_offset:]
    labels = generate_folio_sequence(len(page_paths), start_folio, start_side)
    header_info = read_page_header(transcript_path)

    entries: List[Dict[str, Any]] = []
    blocks: List[np.ndarray] = []
    for idx, (path, fid) in enumerate(zip(page_paths, labels)):
        res = analyze_page(path, fid, header_info, transcript_path)
        res['entry']['pdf_page'] = idx + page_offset + 1
        entries.append(res['entry'])
        blocks.append(res['glyph_features'])
        _notify(progress_cb, 0.4 + (idx + 1) / len(page_paths) * 0.5, f'{fid} analiz edildi ({idx + 1}/{len(page_paths)})', path)

    _notify(progress_cb, 0.92, 'Külliyat genelinde glif kümeleme yapılıyor')
    glyph_global = cluster_corpus_glyphs(blocks)
    for i, e in enumerate(entries):
        e['glyph_signature'] = glyph_global['histograms'].get(i)

    corpus = {
        'name': slug,
        'source_pdf': os.path.basename(pdf_path),
        'total_pdf_pages': len(all_paths),
        'skipped_pages': [os.path.basename(p) for p in cover_paths],
        'glyph_clusters': {k: v for k, v in glyph_global.items() if k != 'histograms'},
        'pages': entries,
    }
    save_corpus(corpus)
    _notify(progress_cb, 1.0, 'Külliyat kaydedildi')
    return corpus


def save_corpus(corpus: Dict[str, Any]) -> str:
    d = os.path.join(CORPUS_ROOT, corpus['name'])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'corpus.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    return path


def list_corpora() -> List[str]:
    if not os.path.isdir(CORPUS_ROOT):
        return []
    return sorted(n for n in os.listdir(CORPUS_ROOT)
                  if os.path.exists(os.path.join(CORPUS_ROOT, n, 'corpus.json')))


def load_corpus(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(CORPUS_ROOT, name, 'corpus.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def corpus_summary(corpus: Dict[str, Any]) -> Dict[str, Any]:
    """Corpus-level aggregate statistics."""
    pages = corpus['pages']
    with_tr = [p for p in pages if p['transcription'].get('available')]
    by_section: Dict[str, int] = {}
    for p in pages:
        by_section[p['section']] = by_section.get(p['section'], 0) + 1
    all_tokens = [t for p in with_tr for t in p['transcription'].get('token_list', [])]
    out: Dict[str, Any] = {
        'pages': len(pages),
        'pages_with_transcription': len(with_tr),
        'by_section': by_section,
        'total_words_detected': sum(p['vision'].get('words', 0) for p in pages),
        'total_tokens_transcribed': len(all_tokens),
        'vocabulary_size': len(set(all_tokens)),
    }
    if all_tokens:
        text = ' '.join(all_tokens)
        ent = compute_shannon_entropy(text)
        out.update({'h1': ent['h1'], 'h2': ent['h2'],
                    'zipf_alpha': analyze_zipf_law(all_tokens)['alpha']})
    return out
