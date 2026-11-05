"""
VoynichLab: Advanced Computational Linguistics & Cryptanalysis Workstation
created by E2
Run with: streamlit run web/app.py
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is in python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.vision_segmenter import load_image
from src.pipeline import run_folio_analysis
from src.pdf_corpus import build_corpus, list_corpora, load_corpus, corpus_summary
from src.zl_corpus import load_zl, normalize_folio_id
from src.slot_grammar import (
    parse_token, compare_grammars_on_tokens,
    GRAMMAR_PROFILES
)
from src.graph_engine import (
    layout_page, draw_overlay, word_crop, align_words_to_tokens,
    find_matching_folios, DEFAULT_PARAMS
)
from src.crypto_studio import (
    SubstitutionCipher, PageAnnotationManager, GlobalLexicon,
    CIPHER_PRESETS, CATEGORIZED_CIPHER_PRESETS, EVA_ALPHABET, MEDIEVAL_BOTANICAL_LEXICON
)
from src.historical_ciphers import (
    CardanGrilleSimulator, HomophonicCipherEngine, PolyalphabeticSolver
)
from src.philology import (
    analyze_hapax_and_mandelbrot, analyze_curio_repetitions,
    analyze_currier_dialect_overlap, compute_slot_transition_matrix
)
from src.crypto_entropy import (
    compute_renyi_entropy, compute_multiscale_shannon_entropy,
    compute_advanced_ic, compute_cross_entropy_and_kl
)
from web.components.editor_iiif import (
    render_transcription_editor, render_openseadragon_iiif_viewer
)
from web.components.e2canvas import render_e2canvas
from src.e2canvas_backend import analyze_canvas_cluster
import base64
from collections import Counter


def get_crop_b64(img: np.ndarray, word_dict: Dict[str, Any]) -> str:
    """Extracts cropped image for word and returns base64 jpeg."""
    if img is None or img.size == 0:
        return ""
    c_img = word_crop(img, word_dict)
    if c_img is not None and c_img.size > 0:
        _, c_buf = cv2.imencode('.jpg', c_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(c_buf).decode('utf-8')
    return ""


def render_word_hover_badge(token: str, crop_b64: str = "", bbox_str: str = "") -> str:
    """Generates an HTML hover badge with miniature image crop preview."""
    if not crop_b64:
        return f'<span class="badge-plain">{token}</span>'
    return (
        f'<span class="crop-hover-container">{token}'
        f'<span class="crop-hover-popup">'
        f'<img src="data:image/jpeg;base64,{crop_b64}" style="max-height: 48px; border-radius: 3px; display: block; margin-bottom: 3px;" />'
        f'<span style="font-size: 0.68rem; color: #a3a3a3; font-family: monospace;">{token} {bbox_str}</span>'
        f'</span></span>'
    )


st.set_page_config(
    page_title="VoynichLab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MATTE BLACK RESEARCH LABORATORY THEME (CSS)
# Zero emojis, high contrast, clean typography, compact layout
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Matte Dark Tone */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #121212 !important;
        color: #e5e5e5 !important;
    }
    code, pre, .stCode, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 98% !important;
    }
    header[data-testid="stHeader"] {
        background-color: #121212 !important;
    }

    /* Sidebar Brand Box */
    .sidebar-brand-box {
        padding: 12px 14px;
        background: #181818;
        border: 1px solid #282828;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #ffffff;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .sidebar-author {
        font-size: 0.74rem;
        color: #888888;
        margin-top: 2px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .sidebar-telemetry {
        border-top: 1px solid #262626;
        padding-top: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #737373;
        line-height: 1.5;
        letter-spacing: 0.2px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161616 !important;
        border-right: 1px solid #262626 !important;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #888888;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    /* Clean Matte Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #161616;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid #262626;
        margin-top: 12px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: 500;
        font-size: 0.84rem;
        color: #888888;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #cccccc;
        background: #1f1f1f;
    }
    .stTabs [aria-selected="true"] {
        background: #242424 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: 1px solid #383838 !important;
    }

    /* Metric Display */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #f5f5f5 !important;
        font-size: 1.35rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Deciphered Text Badges */
    .badge-hit {
        background: #163828;
        color: #86efac;
        border: 1px solid #1f533a;
        padding: 2px 7px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .badge-plain {
        background: #1e1e1e;
        color: #cccccc;
        border: 1px solid #2e2e2e;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
    }
    .decipher-line {
        background: #171717;
        border: 1px solid #282828;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .decipher-eva {
        font-size: 0.74rem;
        color: #737373;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }

    /* Expander and Table Uniformity */
    .streamlit-expanderHeader {
        background-color: #191919 !important;
        border: 1px solid #262626 !important;
        border-radius: 4px !important;
        color: #e5e5e5 !important;
        font-size: 0.86rem !important;
    }
    div[data-testid="stExpander"] {
        border: none !important;
        margin-bottom: 8px;
    }

    /* Word Crop Hover Popover */
    .crop-hover-container {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #181818;
        border: 1px solid #2d2d2d;
        padding: 2px 7px;
        border-radius: 4px;
        cursor: pointer;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        margin: 2px;
        transition: border-color 0.15s ease;
    }
    .crop-hover-container:hover {
        border-color: #38bdf8;
        background: #202020;
    }
    .crop-hover-container .crop-hover-popup {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: #141414;
        border: 1.5px solid #38bdf8;
        padding: 5px;
        border-radius: 5px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.85);
        z-index: 99999;
        transition: opacity 0.18s ease, visibility 0.18s ease;
        pointer-events: none;
        white-space: nowrap;
    }
    .crop-hover-container:hover .crop-hover-popup {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="sidebar-title">VoynichLab</div>
    <div class="sidebar-author">created by E2</div>
    <div class="sidebar-telemetry">
        <div>MOTOR: DSU-GRAPH v2.4</div>
        <div>KULLIYAT: ZL3b-n (39,020 TOKEN)</div>
        <div>MODEL: FINITE-STATE & SHANNON</div>
    </div>
</div>
""", unsafe_allow_html=True)

# e2Canvas Sandbox Mode Button
if 'e2canvas_mode' not in st.session_state:
    st.session_state['e2canvas_mode'] = False

if not st.session_state['e2canvas_mode']:
    if st.sidebar.button("SANDBOX: e2Canvas Modunu Ac", use_container_width=True, type="primary"):
        st.session_state['e2canvas_mode'] = True
        st.rerun()
else:
    if st.sidebar.button("KAPAT: 9 Sekmeli Ana Ekrana Don", use_container_width=True):
        st.session_state['e2canvas_mode'] = False
        st.rerun()

st.sidebar.markdown('<div class="sidebar-section-title">VERI KAYNAGI VE BELGE YONETIMI</div>', unsafe_allow_html=True)

FOLIO_PRESETS = {
    "f11r (Silene Acaulis / Rozet Mavi Cicek)": "f11r",
    "f11v (Leonotus / Cift Kolonlu Govde)": "f11v",
    "f16r (Cannabis / Kenevir Bitkisi)": "f16r",
    "f18r (Cyanus / Mavi Cicek, Kirmizi Merkez)": "f18r"
}

raw_img_dir = os.path.join(base_dir, "data", "raw_images")
upload_dir = os.path.join(base_dir, "data", "uploads")
os.makedirs(upload_dir, exist_ok=True)

source_mode = st.sidebar.radio("Calisma Modu:", ["Tek Sayfa / Ornek Folio", "Tam PDF Kulliyati"])

active_corpus = None
active_page_meta = None

if source_mode == "Tek Sayfa / Ornek Folio":
    selected_label = st.sidebar.selectbox("Katalog Sayfasi Sec:", list(FOLIO_PRESETS.keys()))
    folio_code = FOLIO_PRESETS[selected_label]
    
    with st.sidebar.expander("Ozel Gorsel Yukle (PNG/JPG)", expanded=False):
        custom_upload = st.file_uploader("Yerel Gorsel Dosyasi:", type=["png", "jpg", "jpeg"])
        
    if custom_upload is not None:
        temp_img_path = os.path.join(raw_img_dir, "temp_custom.png")
        with open(temp_img_path, "wb") as f:
            f.write(custom_upload.read())
        active_image_path = temp_img_path
        active_folio_id = "custom"
    else:
        active_image_path = os.path.join(raw_img_dir, f"{folio_code}.png")
        active_folio_id = folio_code
else:
    with st.sidebar.expander("Yeni PDF Kulliyati Yukle", expanded=False):
        pdf_upload = st.file_uploader("Voynich PDF Dosyasi:", type=["pdf"])
        start_folio = st.number_input("Ilk Folio No:", min_value=1, max_value=116, value=1)
        start_side = st.selectbox("Ilk Sayfa Yuzu:", ["r (recto)", "v (verso)"])[0]
        page_offset = st.number_input("Atlanacak Kapak Sayfasi:", min_value=0, max_value=50, value=0)
        
        if pdf_upload is not None and st.button("PDF Ayristir ve Kulliyat Olustur", use_container_width=True):
            pdf_path = os.path.join(upload_dir, pdf_upload.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_upload.getbuffer())
            bar = st.progress(0.0, text="Ayristirma suruyor...")
            preview_placeholder = st.empty()

            def on_progress(p, msg, img_path=None):
                bar.progress(min(p, 1.0), text=msg)
                if img_path and os.path.exists(img_path):
                    preview_placeholder.image(img_path, caption=f"Analiz Edilen: {os.path.basename(img_path)}", use_container_width=True)

            corpus_obj = build_corpus(
                pdf_path, pdf_upload.name, start_folio=int(start_folio), start_side=start_side,
                page_offset=int(page_offset), progress_cb=on_progress
            )
            st.session_state["active_corpus_name"] = corpus_obj["name"]
            st.success(f"{len(corpus_obj['pages'])} sayfa kulliyata kaydedildi.")

    corpora = list_corpora()
    if not corpora:
        st.info("Kayitli kulliyat bulunamadi. Sol panelden bir PDF yukleyin veya 'Tek Sayfa' moduna gecin.")
        st.stop()
    default_idx = corpora.index(st.session_state["active_corpus_name"]) if st.session_state.get("active_corpus_name") in corpora else 0
    corpus_name = st.sidebar.selectbox("Kayitli Kulliyat:", corpora, index=default_idx)
    active_corpus = load_corpus(corpus_name)
    pages = active_corpus["pages"]
    page_labels = [f"{p['folio_id']} - {p['section']}" for p in pages]
    sel_idx = st.sidebar.selectbox("Incelenecek Folio:", range(len(pages)), format_func=lambda i: page_labels[i])
    active_page_meta = pages[sel_idx]
    active_image_path = active_page_meta["image_path"]
    active_folio_id = active_page_meta["folio_id"]

# Sidebar Kaynak Yonetimi (Degistir / Sil)
with st.sidebar.popover("Kaynak Yonetimi (Degistir / Sil)", use_container_width=True):
    st.markdown(f"**Aktif Kaynak:** `{active_folio_id}`")
    st.caption(f"Dosya: `{os.path.basename(active_image_path)}`")
    st.markdown("---")
    st.markdown("**Kaynak Degistir:**")
    replace_file = st.file_uploader("Yerine Yeni Dosya Yukle:", type=["png", "jpg", "jpeg", "pdf"], key="replace_file_uploader")
    if replace_file is not None and st.button("Dosyayi Degistir ve Guncelle", use_container_width=True):
        if replace_file.name.lower().endswith(".pdf"):
            r_pdf_path = os.path.join(upload_dir, replace_file.name)
            with open(r_pdf_path, "wb") as f:
                f.write(replace_file.getbuffer())
            new_c = build_corpus(r_pdf_path, replace_file.name)
            st.session_state["active_corpus_name"] = new_c["name"]
        else:
            cust_p = os.path.join(raw_img_dir, "temp_custom.png")
            with open(cust_p, "wb") as f:
                f.write(replace_file.getbuffer())
        st.rerun()
    st.markdown("---")
    st.markdown("**Kaynak Sil:**")
    if st.button("Aktif Ozel Kaynagi Sil", use_container_width=True):
        cust_p = os.path.join(raw_img_dir, "temp_custom.png")
        if os.path.exists(cust_p):
            try:
                os.remove(cust_p)
            except Exception:
                pass
        st.session_state.pop("active_corpus_name", None)
        st.rerun()

st.sidebar.markdown("---")

# ----------------- RE-ANALYSIS PARAMETERS IN COLLAPSIBLE EXPANDER -----------------
st.sidebar.markdown('<div class="sidebar-section-title">CIZGE ANALIZ AYARLARI</div>', unsafe_allow_html=True)
with st.sidebar.expander("Sayfa Analiz Parametreleri", expanded=False):
    param_sens = st.slider("Murekkep Hassasiyeti", min_value=0.5, max_value=2.0, value=1.0, step=0.05,
                           help="Soluk murekkep vuruslarini yakalama esigi.")
    param_gap = st.slider("Kelime Bosluk Katsayisi", min_value=0.5, max_value=2.0, value=1.0, step=0.05,
                          help="Kelimeleri ayirma hassasiyeti.")
    param_paint = st.slider("Boya ve Pigment Filtresi", min_value=0.1, max_value=0.8, value=0.4, step=0.05,
                            help="Yesil/mavi/kirmizi boya lekelerini yazidan ayirma.")
    param_min_comp = st.slider("Min Satir Bileseni", min_value=1, max_value=6, value=2, step=1)

custom_params = {
    'sensitivity': float(param_sens),
    'word_gap_scale': float(param_gap),
    'paint_reject': float(param_paint),
    'min_line_components': int(param_min_comp)
}

if st.sidebar.button("Sayfayi Tekrar Analiz Et", use_container_width=True):
    st.session_state[f"force_recalc_{active_folio_id}"] = st.session_state.get(f"force_recalc_{active_folio_id}", 0) + 1
    st.session_state[f"params_{active_folio_id}"] = custom_params
    st.sidebar.success("Sayfa yeniden analiz edildi.")

active_custom_params = st.session_state.get(f"params_{active_folio_id}", custom_params)

# Sidebar Folio Metadata Summary
st.sidebar.markdown('<div class="sidebar-section-title">FOLIO TELEMETRISI</div>', unsafe_allow_html=True)
with st.sidebar.expander("Aktif Belge Bilgileri", expanded=True):
    if active_page_meta is not None:
        hdr = active_page_meta.get("header", {})
        st.markdown(f"""
- **Folio Kodu:** `{active_folio_id}`
- **Bolum:** {active_page_meta['section']}
- **Currier Lehcesi:** {hdr.get('language') or 'Bilinmiyor'}
- **Katip Eli:** {hdr.get('hand') or 'Bilinmiyor'}
- **Bitki Kimligi:** `{hdr.get('plant_id') or '-'}`
- **Transkripsiyon:** {'ZL3b-n Mevcut' if active_page_meta['transcription'].get('available') else 'Yok'}
""")
    else:
        zl_all = load_zl()
        clean_id = normalize_folio_id(active_folio_id)
        rec = zl_all.get(clean_id)
        if rec:
            st.markdown(f"""
- **Folio Kodu:** `{active_folio_id}`
- **Bolum:** {rec['section']}
- **Currier Lehcesi:** {rec['language'] or 'A'}
- **Katip Eli:** {rec['hand'] or 'Scribe 1'}
- **Transkripsiyon:** ZL3b-n Extended EVA
""")
        else:
            st.markdown(f"""
- **Folio Kodu:** `{active_folio_id}` (Ozel Yukleme)
- **Bolum:** Otomatik Cizge Esleme
- **Transkripsiyon:** Dinamik Hizalama
""")

# Sidebar Active Folio Image Preview
st.sidebar.markdown('<div class="sidebar-section-title">ANALIZ ONIZLEME EKRANI</div>', unsafe_allow_html=True)
if active_image_path and os.path.exists(active_image_path):
    st.sidebar.image(active_image_path, caption=f"Aktif Sayfa: {active_folio_id}", use_container_width=True)


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
with st.spinner("VoynichLab analiz motorlari calistiriliyor..."):
    analysis = run_folio_analysis(active_image_path, active_folio_id, params=active_custom_params)

orig_img = load_image(active_image_path)
layout = analysis.get('graph_layout', {})
alignment = analysis.get('alignment', {})
word_tokens_map = alignment.get('word_tokens', {})


# ==============================================================================
# e2Canvas SANDBOX WORKSTATION MODE
# ==============================================================================
if st.session_state.get('e2canvas_mode', False):
    top_c1, top_c2 = st.columns([8, 2])
    with top_c1:
        st.markdown("### [SANDBOX MODU: e2Canvas] Sonsuz Kanvas ve Grup Kripto Analizörü")
        st.caption(f"Aktif Folio: {active_folio_id} | Varlık Kütüphanesinden kırpıntı ve kelimeleri tuvale ekleyin, köprülerle (=======>) bağlayın.")
    with top_c2:
        if st.button("X Kapat (Ana Ekrana Don)", use_container_width=True):
            st.session_state['e2canvas_mode'] = False
            st.rerun()

    # Prepare assets for e2Canvas
    canvas_crops = []
    words_list = layout.get('words', [])
    word_tokens_map = alignment.get('word_tokens', {})
    for idx, w in enumerate(words_list[:40]):
        c_img = word_crop(orig_img, w)
        if c_img is not None and c_img.size > 0:
            _, c_buf = cv2.imencode('.jpg', c_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            c_b64 = base64.b64encode(c_buf).decode('utf-8')
            mapped_toks = word_tokens_map.get(idx, {}).get('tokens', [])
            eva_label = mapped_toks[0] if mapped_toks else f"w_{idx+1}"
            canvas_crops.append({
                'id': f"crop_{idx+1}",
                'eva': eva_label,
                'img_b64': c_b64,
                'conf': float(w.get('conf', 0.9)),
                'line': int(w.get('line_id', 1))
            })

    corpus_tokens = [t for l in analysis.get('text_corpus', {}).get('lines', []) for t in l.get('tokens', [])]
    token_counts = Counter(corpus_tokens)
    canvas_words = []
    for t, cnt in token_counts.most_common(50):
        canvas_words.append({
            'eva': t,
            'count': cnt,
            'line': 1,
            'slot': 'core'
        })

    canvas_hypo = [
        {'eva': 'daiin', 'translation': 'folium', 'lang': 'Latince', 'meaning': 'yaprak'},
        {'eva': 'qokedy', 'translation': 'radix', 'lang': 'Latince', 'meaning': 'kök'},
        {'eva': 'chedy', 'translation': 'herba', 'lang': 'Latince', 'meaning': 'ot / bitki'},
        {'eva': 'shedy', 'translation': 'caulis', 'lang': 'Latince', 'meaning': 'gövde / sap'},
        {'eva': 'otchor', 'translation': 'calida', 'lang': 'Latince', 'meaning': 'sıcak mizaç'},
        {'eva': 'chol', 'translation': 'aqua', 'lang': 'Latince', 'meaning': 'su / sıvı'},
        {'eva': 'chor', 'translation': 'oleum', 'lang': 'Latince', 'meaning': 'yağ'},
        {'eva': 'cthaiin', 'translation': 'flores', 'lang': 'Latince', 'meaning': 'çiçekler'},
        {'eva': 'qokain', 'translation': 'fructus', 'lang': 'Latince', 'meaning': 'meyve / tohum'},
        {'eva': 'dal', 'translation': 'semen', 'lang': 'Latince', 'meaning': 'tohum'},
        {'eva': 'chey', 'translation': 'foglia', 'lang': 'İtalyanca', 'meaning': 'yaprak'},
        {'eva': 'okedy', 'translation': 'radice', 'lang': 'İtalyanca', 'meaning': 'kök'},
    ]

    render_e2canvas(active_folio_id, canvas_crops, canvas_words, canvas_hypo, height=760)
    st.stop()


# ==============================================================================
# WORKSTATION TABS (ZERO EMOJIS, HIGH CLARITY)
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1. Gorsel ve Cizge Motoru",
    "2. Transkripsiyon ve Editor",
    "3. Kriptolog Studyosu",
    "4. Tarihsel Kripto Simulatoru",
    "5. Bilgi Teorisi ve Renyi Entropisi",
    "6. Hesaplamali Filoloji",
    "7. Slot Morfolojisi",
    "8. Botanik ve Izomorf Esleme",
    "9. Kulliyat Arsivi"
])


# ==============================================================================
# TAB 1: VISION & GRAPH ENGINE (WITH OPENSEADRAGON IIIF)
# ==============================================================================
with tab1:
    st.subheader("Olcek-Uyarlamali Cizge Motoru ve IIIF Deep-Zoom Tuvali")
    
    view_mode = st.radio("Goruntuleme Modu:", ["Standart Segmentasyon Katmanlari", "OpenSeadragon IIIF Deep-Zoom Tuvali"], horizontal=True)

    if view_mode == "OpenSeadragon IIIF Deep-Zoom Tuvali":
        st.caption("Yale Beinecke MS 408 yuksek cozunurluklu gorselinde Annotorious vektor katmani: Kutu araci (B), Cokgen araci (P), Secim (S), 8-yonlu tutamaclar ve popover denetci:")
        raw_words = layout.get('words', [])
        enriched_words = []
        for i, w in enumerate(raw_words):
            w_copy = dict(w)
            mapped_toks = word_tokens_map.get(i, {}).get('tokens', [])
            if mapped_toks:
                w_copy['eva'] = mapped_toks[0]
            enriched_words.append(w_copy)
        render_openseadragon_iiif_viewer(orig_img, enriched_words, height=580, folio_id=active_folio_id)
    else:
        st.markdown("**Gorsel Katman Denetimi:**")
        lcol1, lcol2, lcol3, lcol4, lcol5, lcol6 = st.columns(6)
        with lcol1:
            layer_words = st.checkbox("Kelimeler", value=True)
        with lcol2:
            layer_lines = st.checkbox("Satir Baselines", value=True)
        with lcol3:
            layer_blocks = st.checkbox("Paragraf Bloklari", value=True)
        with lcol4:
            layer_graph = st.checkbox("Cizge Kenarlari (DSU)", value=False)
        with lcol5:
            layer_comps = st.checkbox("Murekkep Vuruslari", value=False)
        with lcol6:
            layer_order = st.checkbox("Okuma Siralamasi", value=False)

        layers_dict = {
            'words': layer_words, 'lines': layer_lines, 'blocks': layer_blocks,
            'graph': layer_graph, 'components': layer_comps, 'order': layer_order
        }

        overlay_img = draw_overlay(orig_img, layout, layers=layers_dict)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), caption=f"Orijinal El Yazmasi ({active_folio_id})", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption="Graph Engine Segmentasyon Analizi", use_container_width=True)

    # Metrics HUD
    stats = layout.get('stats', {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Glif Olcegi (m)", f"{layout.get('m', 0.0):.1f} px")
    m2.metric("Tespit Edilen Kelime", stats.get('words', 0))
    m3.metric("Mantiksal Satir", stats.get('logical_lines', 0))
    m4.metric("Paragraf Blogu", stats.get('blocks', 0))
    m5.metric("Kelime Bosluk Esigi", f"{stats.get('word_gap_threshold', 0.0):.1f} px")

    st.markdown("---")
    st.markdown("**Cizge ve Boya Ayristirma Telemetrisi:**")
    t1, t2, t3, t4 = st.columns(4)
    t1.write(f"- Murekkep Vurus Bileseni: `{stats.get('components', 0)}`")
    t2.write(f"- Filtrelenen Boya Lekesi: `{stats.get('rejected_paint', 0)}`")
    t3.write(f"- Asiri Buyuk Sekiller: `{stats.get('rejected_large', 0)}`")
    t4.write(f"- Ortalama Kelime Guveni: `%{stats.get('mean_word_conf', 0.0) * 100:.1f}`")

    cand_folios = analysis.get('candidate_folios', [])
    if cand_folios:
        st.markdown("**Cizge Profili ile Eslesen Voynich Sayfalari (Layout Fingerprint):**")
        cand_df = pd.DataFrame(cand_folios)
        st.dataframe(cand_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: TRANSCRIPTION & CODEMIRROR EDITOR
# ==============================================================================
with tab2:
    st.subheader("Transkripsiyon ve CodeMirror 6 Editoru")
    st.write("Ozel Voynich transkripsiyon dilleri (Extended EVA, Currier, v101, Takahashi) icin syntax highlighting ve autocompletion destekli arastirma editoru:")

    ed_mode = st.selectbox("Transkripsiyon Notasyon Sistemi:", ["EVA (Extensible Voynich Alphabet)", "Currier Notasyonu", "v101 (First Study Group)", "Takahashi"])

    # Prepare raw text from transcription lines
    raw_lines_text = "\n".join([f"<{active_folio_id}.{l['line_number']}> {l['clean_text']}" for l in analysis['text_corpus']['lines']]) or f"<{active_folio_id}.1> daiin qokedy chedy"

    # Render CodeMirror editor component
    render_transcription_editor(raw_lines_text, mode=ed_mode, height=360)

    st.markdown("---")
    st.subheader("Konsensus Transkripsiyon ve DP Hizalamasi")
    if alignment.get('word_tokens'):
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Hizalanan Kelime Kapsami", f"{alignment.get('coverage', 0.0) * 100:.1f}%")
        ac2.metric("Ortalama Hizalama Kalitesi", f"{alignment.get('quality', 0.0) * 100:.1f}%")
        ac3.metric("Olcek (Piksel / Glif Birimi)", f"{alignment.get('scale_px_per_unit', 0.0):.2f} px")

    lines = analysis['text_corpus']['lines']
    words_list = layout.get('words', [])
    word_mapping = alignment.get('word_tokens', {})
    word_b64_lookup = {widx: get_crop_b64(orig_img, wobj) for widx, wobj in enumerate(words_list[:120])}

    if lines:
        for l in lines:
            with st.expander(f"Satir {l['line_number']}: {l['clean_text'][:70]}...", expanded=False):
                hover_badges = []
                for tok in l['tokens']:
                    b64_img = ""
                    b_str = ""
                    for widx, wobj in enumerate(words_list):
                        if tok in word_mapping.get(widx, {}).get('tokens', []):
                            b64_img = word_b64_lookup.get(widx, "")
                            b_str = f"bbox: {wobj['bbox']}"
                            break
                    hover_badges.append(render_word_hover_badge(tok, b64_img, b_str))
                st.markdown(" ".join(hover_badges), unsafe_allow_html=True)
                st.code(l['clean_text'], language="text")
                st.caption(f"Kelime Sayisi: {len(l['tokens'])} | Tokenler: {', '.join(l['tokens'])}")

    st.markdown("---")
    search_token = st.text_input("Sayfa icinde kelime ara (Orn: daiin, qokedy, chol):", value="daiin")
    if search_token:
        all_toks = [t for l in lines for t in l['tokens']]
        matches = [t for t in all_toks if search_token.lower() in t.lower()]
        st.write(f"Bulunan eslesme sayisi: **{len(matches)}** / Toplam {len(all_toks)} kelime")


# ==============================================================================
# TAB 3: CRYPTOLOGIST STUDIO & REAL-TIME SUBSTITUTION WORKBENCH
# ==============================================================================
with tab3:
    st.subheader("Kriptolog Studyosu ve Dahili Ikame Calisma Tahtasi")
    st.write("EVA gliflerine karsilik atandiginda tum sayfada aninda guncellenen cozumleme ve hedef dil cross-entropy skorlama modulu:")

    annot_mgr = PageAnnotationManager(active_folio_id)
    global_lex = GlobalLexicon()

    subtab_words, subtab_cipher, subtab_lexicon = st.tabs([
        "Kelime ve Glif Duzenleyici",
        "Kripto Ikame Masasi",
        "Kulliyat Sozlugu ve Veriler"
    ])

    # ---------------- SUBTAB A: WORD INSPECTOR & EDITOR ----------------
    with subtab_words:
        st.markdown("#### Sayfa Kelimesi Inceleme ve Duzenleme (Envanter Secim Paneli)")
        words_list = layout.get('words', [])
        word_mapping = alignment.get('word_tokens', {})

        if not words_list:
            st.info("Bu sayfada tespit edilen kelime bulunamadi.")
        else:
            inv_col1, inv_col2 = st.columns([1, 2])
            with inv_col1:
                inv_mode = st.radio(
                    "Envanter Kaynagi:",
                    ["Sayfa Kelimeleri", "Kulliyat Frekansi", "Manuel Koordinat"],
                    horizontal=False
                )

            selected_word = words_list[0]
            w_idx = 0
            default_reading = ""

            with inv_col2:
                if inv_mode == "Sayfa Kelimeleri":
                    options = []
                    for i, w in enumerate(words_list):
                        mapped = word_mapping.get(i, {}).get('tokens', [])
                        lbl = mapped[0] if mapped else f"w_{i+1}"
                        options.append(f"#{i+1} | {lbl} (Satir {w.get('line_id', 1)}, bbox: {w['bbox']})")
                    w_idx = st.selectbox("Incelenecek Kelime:", range(len(options)), format_func=lambda i: options[i])
                    selected_word = words_list[w_idx]
                    aligned_data = word_mapping.get(w_idx, {})
                    candidate_tokens = aligned_data.get('tokens', [])
                    default_reading = candidate_tokens[0] if candidate_tokens else ""
                elif inv_mode == "Kulliyat Frekansi":
                    corpus_tokens = [tok for l in analysis['text_corpus']['lines'] for tok in l['tokens']]
                    top_tokens = [t for t, _ in Counter(corpus_tokens).most_common(40)]
                    sel_tok = st.selectbox("Kulliyat Tokeni Sec:", top_tokens if top_tokens else ["daiin"])
                    default_reading = sel_tok
                    matching_indices = [i for i, w in enumerate(words_list) if sel_tok in word_mapping.get(i, {}).get('tokens', [])]
                    w_idx = matching_indices[0] if matching_indices else 0
                    selected_word = words_list[w_idx]
                else:
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    with m_c1:
                        bx = st.number_input("X:", min_value=0, max_value=orig_img.shape[1], value=int(words_list[0]['bbox'][0]))
                    with m_c2:
                        by = st.number_input("Y:", min_value=0, max_value=orig_img.shape[0], value=int(words_list[0]['bbox'][1]))
                    with m_c3:
                        bw = st.number_input("W:", min_value=5, max_value=orig_img.shape[1], value=int(words_list[0]['bbox'][2]))
                    with m_c4:
                        bh = st.number_input("H:", min_value=5, max_value=orig_img.shape[0], value=int(words_list[0]['bbox'][3]))
                    selected_word = {'bbox': [int(bx), int(by), int(bw), int(bh)], 'conf': 0.95, 'line_id': 1}
                    w_idx = 0
                    default_reading = ""

            cropped = word_crop(orig_img, selected_word)

            wc_col1, wc_col2 = st.columns([1, 2])
            with wc_col1:
                st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption=f"Secili Kelime {selected_word['bbox']}", use_container_width=True)

            aligned_data = word_mapping.get(w_idx, {})
            if not default_reading:
                candidate_tokens = aligned_data.get('tokens', [])
                default_reading = candidate_tokens[0] if candidate_tokens else ""

            saved_annot = annot_mgr.get_word_annotation(w_idx) or {}
            saved_reading = saved_annot.get('reading', default_reading)
            saved_meaning = saved_annot.get('meaning', '')
            saved_pos = saved_annot.get('pos', 'Isim')
            saved_notes = saved_annot.get('notes', '')
            saved_conf = saved_annot.get('confidence', 3)

            slot_info = parse_token(saved_reading) if saved_reading else {}

            with wc_col2:
                st.markdown(f"**Konsensus EVA Okumasi:** `{default_reading or 'Tespit Edilemedi'}`")
                if slot_info.get('is_valid'):
                    st.caption(f"Morfolojik Slot: Onek: {slot_info.get('prefix') or '-'} | Gallows: {slot_info.get('medial') or '-'} | Cekirdek: {slot_info.get('core') or '-'} | Sonek: {slot_info.get('suffix') or '-'}")

                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    user_reading = st.text_input("Kendi Okumaniz / Alternatif Transkripsiyon:", value=saved_reading)
                    user_meaning = st.text_input("Hipotez Anlami / Ceviri (Orn: folium / yaprak):", value=saved_meaning)
                with e_col2:
                    pos_options = ["Isim", "Fiil", "Sifat", "Belirtec", "Bitki Adi", "Kozmolojik", "Onek / Morfem", "Sayi / Miktar", "Diger"]
                    pos_idx = pos_options.index(saved_pos) if saved_pos in pos_options else 0
                    user_pos = st.selectbox("Dilbilgisel Rol (POS):", pos_options, index=pos_idx)
                    user_conf = st.slider("Guven Derecesi (1-5):", min_value=1, max_value=5, value=saved_conf)

                user_notes = st.text_area("Kriptolog Notu / Gozlemler:", value=saved_notes, height=68)

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Bu Kelimeyi Kaydet (Folio Seviyesi)", use_container_width=True):
                        annot_mgr.set_word_annotation(
                            word_idx=w_idx, reading=user_reading, meaning=user_meaning,
                            pos=user_pos, slot=slot_info, notes=user_notes, confidence=user_conf
                        )
                        st.success(f"Kelime #{w_idx} kaydedildi.")
                with btn_col2:
                    if st.button("Kulliyat Sozlugune Ekle (Genel)", use_container_width=True):
                        annot_mgr.set_word_annotation(
                            word_idx=w_idx, reading=user_reading, meaning=user_meaning,
                            pos=user_pos, slot=slot_info, notes=user_notes, confidence=user_conf
                        )
                        global_lex.add_entry(
                            token_eva=user_reading, meaning=user_meaning, pos=user_pos,
                            source_folio=active_folio_id, notes=user_notes
                        )
                        st.success(f"'{user_reading}' genel sozluge eklendi.")

        saved_words = annot_mgr.data.get('word_annotations', {})
        if saved_words:
            st.markdown("---")
            st.markdown(f"**Bu Sayfada Kaydedilen Aciklamalar ({len(saved_words)} Kelime):**")
            ann_rows = []
            for wid, dat in sorted(saved_words.items(), key=lambda x: int(x[0])):
                ann_rows.append({
                    "Kelime #": int(wid),
                    "Okuma": dat.get('reading', ''),
                    "Anlam": dat.get('meaning', ''),
                    "Dilbilgisel Rol": dat.get('pos', ''),
                    "Guven": f"{dat.get('confidence', 3)}/5",
                    "Not": dat.get('notes', '')
                })
            st.dataframe(pd.DataFrame(ann_rows), use_container_width=True, hide_index=True)

    # ---------------- SUBTAB B: REAL-TIME CIPHER LABORATORY ----------------
    with subtab_cipher:
        st.markdown("#### Gercek Zamanli Kripto Ikame Masasi")
        
        target_lang = st.selectbox("Hedef Dil Karsilastirmasi:", ["latin", "italian", "greek", "hebrew"], format_func=lambda s: {"latin": "Latince (Klasik)", "italian": "Erken Italyanca", "greek": "Grekce", "hebrew": "Ibranice"}.get(s, s))

        if "cipher_mapping" not in st.session_state:
            st.session_state["cipher_mapping"] = dict(CIPHER_PRESETS["Sıfırla (Birebir EVA)"])

        # Categorized preset selection
        st.markdown("**Tarihsel Dil ve Sifreleme Hipotezi Sec:**")
        p_c1, p_c2, p_c3, p_c4 = st.columns([3, 4, 2, 2])
        with p_c1:
            sel_cat = st.selectbox("Dil Kategorisi:", list(CATEGORIZED_CIPHER_PRESETS.keys()))
        with p_c2:
            sel_lang = st.selectbox("Hedef Dil / Alfabe:", list(CATEGORIZED_CIPHER_PRESETS[sel_cat].keys()))
        with p_c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Masaya Yukle", use_container_width=True):
                st.session_state["cipher_mapping"] = dict(CATEGORIZED_CIPHER_PRESETS[sel_cat][sel_lang])
                st.rerun()
        with p_c4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Sifirla (EVA)", use_container_width=True):
                st.session_state["cipher_mapping"] = dict(CIPHER_PRESETS["Sıfırla (Birebir EVA)"])
                st.rerun()

        st.markdown("**Glif Ikame Tablosu (EVA -> Hedef Harf):**")
        cur_map = st.session_state["cipher_mapping"]
        cols_per_row = 10
        row1_glyphs = EVA_ALPHABET[:10]
        row2_glyphs = EVA_ALPHABET[10:]

        c_row1 = st.columns(cols_per_row)
        for i, g in enumerate(row1_glyphs):
            with c_row1[i]:
                new_val = st.text_input(f"{g} ->", value=cur_map.get(g, g), key=f"glyph_{g}", max_chars=3)
                cur_map[g] = new_val.strip()

        c_row2 = st.columns(cols_per_row)
        for i, g in enumerate(row2_glyphs):
            with c_row2[i]:
                new_val = st.text_input(f"{g} ->", value=cur_map.get(g, g), key=f"glyph_{g}", max_chars=3)
                cur_map[g] = new_val.strip()

        active_tokens = [tok for l in analysis['text_corpus']['lines'] for tok in l['tokens']]
        cipher = SubstitutionCipher(cur_map)
        eval_metrics = cipher.evaluate_decipherment(active_tokens, target_lang=target_lang)

        st.markdown("---")
        st.markdown("**Kriptografik Skor ve Cross-Entropy Uyum Tablosu:**")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Sozluk Eslesme Orani", f"%{eval_metrics['hit_rate']}")
        sc2.metric("Sartli Entropi (H2)", f"{eval_metrics['entropy_h2']} bit")
        sc3.metric("Cross-Entropy H(P,Q)", f"{eval_metrics['cross_entropy']} bit")
        sc4.metric("KL Divergence", f"{eval_metrics['kl_divergence']}")
        sc5.metric("Kripto Uyumluluk Skoru", f"%{eval_metrics['plausibility_score']}")

        if eval_metrics['matched_words']:
            st.success(f"Eslesen Orta Cag Botanik/Tibbi Kelimeleri: {', '.join([m['deciphered'] for m in eval_metrics['matched_words']])}")

        st.markdown("---")
        st.markdown("**Cozulmus Metin Onizlemesi (Satir Satir):**")
        for l in analysis['text_corpus']['lines']:
            orig_toks = l['tokens']
            dec_toks = cipher.decipher_tokens(orig_toks)
            
            line_html = []
            for orig, dec in zip(orig_toks, dec_toks):
                if dec.lower() in MEDIEVAL_BOTANICAL_LEXICON:
                    line_html.append(f'<span class="badge-hit">{dec}</span>')
                else:
                    line_html.append(f'<span class="badge-plain">{dec}</span>')
            
            st.markdown(f'<div class="decipher-line"><strong>Satir {l["line_number"]}:</strong> ' + " ".join(line_html) + f'<div class="decipher-eva">EVA: {l["clean_text"]}</div></div>', unsafe_allow_html=True)

    # ---------------- SUBTAB C: GLOBAL LEXICON & EXPORT ----------------
    with subtab_lexicon:
        st.markdown("#### Kulliyat Sozlugu ve Hipotez Arsivi")
        st.write("Arastirmaci tarafindan tespit edilen ve tum kulliyata yayilan sozluk kayitlari:")

        all_entries = global_lex.get_all()
        if not all_entries:
            st.info("Henuz kayitli kelime bulunmuyor. 'Kelime ve Glif Duzenleyici' sekmesinden kelime kaydedebilirsiniz.")
        else:
            lex_rows = []
            for tok, edat in all_entries.items():
                lex_rows.append({
                    "EVA Kelimesi": tok,
                    "Hipotez Anlami": edat.get('meaning', ''),
                    "Dilbilgisel Rol": edat.get('pos', ''),
                    "Kaynak Folyolar": ", ".join(edat.get('folios', [])),
                    "Notlar": edat.get('notes', '')
                })
            lex_df = pd.DataFrame(lex_rows)
            st.dataframe(lex_df, use_container_width=True, hide_index=True)

            lex_col1, lex_col2 = st.columns(2)
            with lex_col1:
                st.download_button(
                    "Sozlugu CSV Olarak Indir",
                    data=lex_df.to_csv(index=False).encode('utf-8'),
                    file_name="voynich_lexicon.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with lex_col2:
                st.download_button(
                    "Sozlugu JSON Olarak Indir",
                    data=json.dumps(all_entries, ensure_ascii=False, indent=2).encode('utf-8'),
                    file_name="voynich_lexicon.json",
                    mime="application/json",
                    use_container_width=True
                )


# ==============================================================================
# TAB 4: HISTORICAL CIPHER SIMULATOR (CARDAN, HOMOPHONIC, ALBERTI/VIGENERE)
# ==============================================================================
with tab4:
    st.subheader("Tarihsel Kripto Hipotez Simulatoru")
    st.write("Ronesans donemi Cardano Izgarasi (Cardan Grille), Homofonik ikame siniflari ve Alberti carki / Vigenere polialfabetik Kasiski muayenesi:")

    h_sub1, h_sub2, h_sub3 = st.tabs([
        "Cardan Grille (Izgara Filtresi)",
        "Homofonik Ikame Motoru",
        "Alberti Carki ve Vigenere Cozucu"
    ])

    hist_data = analysis.get('historical_ciphers', {})

    with h_sub1:
        st.markdown("#### Cardan Grille (Cardano Izgara Maskesi)")
        st.write("K x K sanal 4 rotasyonlu izgara ile sayfa karakter matrisi taranir ve anlamli harf dizileri aranir.")

        cg_col1, cg_col2 = st.columns([1, 2])
        with cg_col1:
            grid_k = st.selectbox("Izgara Boyutu (K x K):", [4, 6, 8], index=1)
            n_trials = st.slider("Simulasyon Deneme Sayisi:", min_value=5, max_value=50, value=20)
            if st.button("Cardan Izgara Taramasini Calistir", use_container_width=True):
                full_raw_text = ' '.join([tok for l in analysis['text_corpus']['lines'] for tok in l['tokens']])
                sim_res = CardanGrilleSimulator(k=grid_k).simulate_search(full_raw_text, n_trials=n_trials)
                st.session_state["cardan_res"] = sim_res

        cardan_res = st.session_state.get("cardan_res", hist_data.get('cardan', {}))
        with cg_col2:
            if cardan_res:
                st.write(f"- En Iyi Simulasyon Skoru: `{cardan_res.get('score', 0.0)}`")
                st.write(f"- Vokal Orani: `%{cardan_res.get('vowel_ratio', 0.0)}`")
                st.write(f"- Eslesen Kokler: `{', '.join(cardan_res.get('matched_roots', [])) or 'Eslesme Yok'}`")
                st.markdown("**Cekilen Ornek Harf Dizisi:**")
                st.code(cardan_res.get('extracted_sample', '')[:100], language="text")

                rounds = cardan_res.get('rounds', {})
                if rounds:
                    r_df = pd.DataFrame([
                        {"Aci": "0° (Kuzey)", "Cekilen Metin": rounds.get('0_deg', '')[:50]},
                        {"Aci": "90° (Dogu)", "Cekilen Metin": rounds.get('90_deg', '')[:50]},
                        {"Aci": "180° (Guney)", "Cekilen Metin": rounds.get('180_deg', '')[:50]},
                        {"Aci": "270° (Bati)", "Cekilen Metin": rounds.get('270_deg', '')[:50]}
                    ])
                    st.dataframe(r_df, use_container_width=True, hide_index=True)

    with h_sub2:
        st.markdown("#### Homofonik / Verbose Ikame Motoru")
        st.write("Bir acik metin harfinin birden fazla Voynich glifine/tokenine donustugu hipotezi icin baglamsal benzerlik kumesi:")
        homo_data = hist_data.get('homophonic', {})
        st.write(f"- Temel Karakter Entropisi: `{homo_data.get('baseline_char_entropy', 0.0)} bit`")
        st.write(f"- Potansiyel Entropi Dususu: `{homo_data.get('entropy_reduction', 0.0)} bit`")
        
        candidates = homo_data.get('homophonic_candidates', [])
        if candidates:
            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
        else:
            st.info("Bu sayfada homofonik aday eslesmesi bulunamadi.")

    with h_sub3:
        st.markdown("#### Alberti Carki ve Vigenere Polialfabetik Cozucu")
        kasiski = hist_data.get('kasiski', {})
        auto = hist_data.get('autocorrelation', {})

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**Kasiski Muayenesi (N-gram Mesafe EBOB Analizi):**")
            probable_keys = kasiski.get('probable_key_lengths', [])
            if probable_keys:
                st.dataframe(pd.DataFrame(probable_keys), use_container_width=True, hide_index=True)
            else:
                st.write("Belirgin periyodik n-gram mesafesi bulunamadi.")

        with col_k2:
            st.markdown("**Friedman Otokorelasyon Testi:**")
            st.write(f"- Ortalama Cakisma Orani: `{auto.get('mean_coincidence', 0.0)}`")
            st.write(f"- Periyodik Zirveler: `{auto.get('periodic_peaks', []) or 'Periyodik Zirve Yok'}`")
            if auto.get('shifts'):
                auto_df = pd.DataFrame(auto['shifts'])
                st.bar_chart(auto_df.set_index('shift'))


# ==============================================================================
# TAB 5: INFORMATION THEORY & RENYI ENTROPY
# ==============================================================================
with tab5:
    st.subheader("Bilgi Teorisi, Renyi Entropisi ve Index of Coincidence")
    crypto = analysis['cryptanalysis']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Derece Entropi (H1)", f"{crypto['shannon_h1']} bit")
    c2.metric("Sartli Entropi (H2)", f"{crypto['shannon_h2']} bit")
    c3.metric("Index of Coincidence (IoC)", f"{crypto['index_of_coincidence']}")
    c4.metric("Zipf Katsayisi (alfa)", f"{crypto['zipf_alpha']}")

    st.markdown("---")
    ren_col1, ren_col2 = st.columns(2)
    with ren_col1:
        st.markdown("#### Renyi Genellestirilmis Entropi Spektrumu")
        st.caption("Renyi entropisi H_alpha, alfa degisirken bilgi dagiliminin carpikligini gosterir (H_0 >= H_1 >= H_2 >= H_inf).")
        renyi_dict = crypto.get('renyi', {})
        if renyi_dict:
            ren_df = pd.DataFrame(list(renyi_dict.items()), columns=["Mertebe (alfa)", "Entropi (bit)"])
            st.dataframe(ren_df, use_container_width=True, hide_index=True)

    with ren_col2:
        st.markdown("#### Index of Coincidence (IoC) Siniflandirmasi")
        adv_ic = crypto.get('advanced_ic', {})
        st.write(f"- **Hesaplanan IoC:** `{adv_ic.get('ic', 0.0)}`")
        st.write(f"- **Normalize IoC (NIC):** `{adv_ic.get('normalized_ic', 0.0)}`")
        st.write(f"- **Karar:** **{adv_ic.get('verdict', '')}**")
        st.caption(adv_ic.get('hypothesis_note', ''))

    st.markdown("---")
    st.markdown("#### Cok Olcekli Entropi ve Diller Arasi Karsilastirma")
    ms_entropy = crypto.get('multiscale', {})
    comp = ms_entropy.get('comparisons', {})
    if comp:
        comp_df = pd.DataFrame.from_dict(comp, orient='index')
        st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")
    vowels = crypto['sukhotin_detected_vowels']
    st.write(f"Sukhotin Algoritmasi Vokal Adaylari: **{', '.join(vowels).upper()}**")
    curr = crypto['currier_dialect']
    st.write(f"Currier Lehcesi: **{curr['dialect']}** (Guven: %{curr['confidence'] * 100:.1f})")


# ==============================================================================
# TAB 6: COMPUTATIONAL PHILOLOGY (HAPAX, ZIPF-MANDELBROT, CURIO REPEATS)
# ==============================================================================
with tab6:
    st.subheader("Hesaplamali Filoloji ve Dilbilim Analizoru")
    phil = analysis.get('philology', {})

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### Hapax Legomena ve Zipf-Mandelbrot Modeli")
        hm = phil.get('hapax_mandelbrot', {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Hapax Sayisi (f=1)", hm.get('hapax_count', 0))
        m2.metric("Hapax Orani (Kelime)", f"%{hm.get('hapax_ratio_token', 0.0) * 100:.1f}")
        m3.metric("Dis Legomena (f=2)", hm.get('dis_legomena_count', 0))

        st.write(f"- **Mandelbrot alfa (egim):** `{hm.get('mandelbrot_alpha', 0.0)}`")
        st.write(f"- **Mandelbrot beta (kayma):** `{hm.get('mandelbrot_beta', 0.0)}`")
        st.write(f"- **Heaps Yasasi Buyume Katsayisi:** `{hm.get('heaps_growth_exponent', 0.0)}`")
        st.write(f"- Ornek Hapax Tokenleri: `{', '.join(hm.get('sample_hapax', [])) or '-'}`")

    with p_col2:
        st.markdown("#### Curio Tekrarlari (Immediate Word Echoes)")
        st.caption("Voynich metninde ard arda gelen ayni kelime tekrarlari dogal dillerden anomalik sekilde 3-5 kat fazladir.")
        cr = phil.get('curio_repetitions', {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Ardisik Tekrar Sayisi", cr.get('immediate_repeats_count', 0))
        c2.metric("Tekrarlanma Orani", f"%{cr.get('duplication_rate_percent', 0.0)}")
        c3.metric("Dogal Dil Anomali Faki", f"{cr.get('anomaly_factor_vs_natural', 1.0)}x")

        top_dups = cr.get('top_duplicated_tokens', [])
        if top_dups:
            st.dataframe(pd.DataFrame(top_dups), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 7: SLOT MORPHOLOGY
# ==============================================================================
with tab7:
    st.subheader("Kati Morfolojik Slot Grameri ve Gecis Matrisi")
    st.markdown("""
    Voynich metninde kelimeler rastgele harf dizilimleri degildir. Kati bir **Slot Grameri**ne uyar:
    $$\\text{Kelime} = [\\text{Onek}] + [\\text{Gallows / Medial}] + [\\text{Cekirdek Vokal}] + [\\text{Sonek}]$$
    """)

    corpus_tokens = [tok for l in analysis['text_corpus']['lines'] for tok in l['tokens']]

    st.markdown("#### Gramer Modeli ve Coklu Profil Karsilastirmasi")
    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        grammar_options = list(GRAMMAR_PROFILES.keys()) + ["Ozel / Kullanici Tanimli Gramer"]
        selected_grammar = st.selectbox("Aktif Slot Grameri Modeli:", grammar_options)
    
    custom_grammar_dict = None
    if selected_grammar == "Ozel / Kullanici Tanimli Gramer":
        with st.expander("Ozel Gramer Bilesenlerini Duzenle", expanded=True):
            cg1, cg2 = st.columns(2)
            with cg1:
                c_pfx = st.text_input("Onekler (virgulle):", value="q, y, o, ch, sh, d, s")
                c_med = st.text_input("Gallows / Medial (virgulle):", value="t, k, p, f")
            with cg2:
                c_cor = st.text_input("Cekirdek Vokal (virgulle):", value="a, e, o, y, ai, ey, ee")
                c_sfx = st.text_input("Sonekler (virgulle):", value="y, l, r, m, n, in, iin, dy")
            custom_grammar_dict = {
                'name': 'Ozel Gramer',
                'description': 'Kullanici tanimli kurallar',
                'prefixes': [s.strip() for s in c_pfx.split(',') if s.strip()],
                'medials': [s.strip() for s in c_med.split(',') if s.strip()],
                'cores': [s.strip() for s in c_cor.split(',') if s.strip()],
                'suffixes': [s.strip() for s in c_sfx.split(',') if s.strip()],
            }

    # Comparative evaluation table across models
    comp_res = compare_grammars_on_tokens(corpus_tokens, list(GRAMMAR_PROFILES.keys()), custom_grammar_dict)
    comp_df = pd.DataFrame(comp_res.get('results', []))
    st.markdown("**Gramer Modelleri Karsilastirma Tablosu (Sayfa Tokenleri Uzerinde):**")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    morph = analysis['morphology']
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Incelenen Token", morph['total_tokens'])
    c2.metric("Kurallara Birebir Uyan", morph['valid_tokens'])
    c3.metric("Gramer Uyum Orani", f"{morph['valid_rate'] * 100:.1f}%")

    st.markdown("#### Onek -> Cekirdek Markov Gecis Olasiliklari")
    slot_trans = analysis.get('philology', {}).get('slot_transitions', {})
    p_to_c = slot_trans.get('prefix_to_core_transitions', [])
    if p_to_c:
        st.dataframe(pd.DataFrame(p_to_c), use_container_width=True, hide_index=True)

    st.markdown("#### Ornek Kelimelerin Slot Ayristirmasi")
    samples_df = pd.DataFrame([
        {
            "Kelime": s['raw'],
            "Onek": s['prefix'] or "-",
            "Gallows/Medial": s['medial'] or "-",
            "Cekirdek": s['core'] or "-",
            "Sonek": s['suffix'] or "-",
            "Gramer Uyumu": "Uyumlu" if s['is_valid'] else "Uyumsuz"
        }
        for s in morph['parsed_samples'][:20]
    ])
    st.dataframe(samples_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 8: BOTANICAL CRIB MATCHING
# ==============================================================================
with tab8:
    st.subheader("Botanik Illustrasyon Crib Esleme ve Kriptografik Sablonlar")
    cribs = analysis['crib_matching']

    st.write(f"Bu sayfadaki bitki cizimi (**{active_folio_id}**) icin Ortacag kaynaklarinda (Dioscorides, Herbarius Moguntinus, Circa Instans) aday gosterilen turler:")
    
    if cribs['botanical_candidates']:
        for b in cribs['botanical_candidates']:
            st.markdown(f"- **{b['name']}** (Latince: *{b['latin']}*, Aile: *{b['family']}*)")
    else:
        st.info("Bu sayfa icin dogrudan botanik crib kaydi bulunamadi.")

    st.markdown("---")
    st.markdown("#### Izomorfik Harf Tekrar Deseni (Letter Pattern Matching)")
    st.caption("Ornek: 'cannabis' kelimesi 0.1.2.2.1.3.4 harf tekrar desenine sahiptir. Sayfa uzerindeki tokenlerin harf tekrar desenleriyle kiyaslama:")

    iso_matches = cribs['isomorphic_matches']
    if iso_matches:
        st.dataframe(pd.DataFrame(iso_matches), use_container_width=True, hide_index=True)
    else:
        st.write("Bu sayfadaki kelimeler arasinda taranan isimlerle birebir izomorfik sablon eslesmesi bulunamadi.")


# ==============================================================================
# TAB 9: CORPUS ARCHIVE
# ==============================================================================
with tab9:
    st.subheader("Kulliyat (Folio Arsivi)")
    if active_corpus is None:
        st.info("Kulliyat gorunumu icin sol panelden 'Tam PDF Kulliyati' calisma modunu secin.")
    else:
        summ = corpus_summary(active_corpus)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Folio Sayisi", summ['pages'])
        c2.metric("Transkripsiyonu Olan", summ['pages_with_transcription'])
        c3.metric("Gorselden Bulunan Kelime", summ['total_words_detected'])
        c4.metric("Sozluk Boyutu", summ['vocabulary_size'])
        if 'h2' in summ:
            d1, d2, d3 = st.columns(3)
            d1.metric("Kulliyat H1", f"{summ['h1']} bit")
            d2.metric("Kulliyat H2", f"{summ['h2']} bit")
            d3.metric("Zipf alfa", summ['zipf_alpha'])
        st.write("**Bolumlere gore dagilim:**")
        st.bar_chart(pd.Series(summ['by_section']))

        rows = []
        for p in active_corpus['pages']:
            tr = p['transcription']
            rows.append({
                'Folio': p['folio_id'], 'PDF Sayfa': p.get('pdf_page'), 'Bolum': p['section'],
                'Lehce (ZL)': p.get('header', {}).get('language', ''),
                'Satir (gorsel)': p['vision'].get('lines'), 'Kelime (gorsel)': p['vision'].get('words'),
                'Token (metin)': tr.get('tokens'), 'H2': tr.get('h2'), 'Zipf alfa': tr.get('zipf_alpha'),
                'Slot uyumu': tr.get('slot_valid_rate'), 'Currier tahmini': tr.get('currier_estimate'),
                'Bitki ID': p.get('header', {}).get('plant_id', ''),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Kulliyat Tablosunu CSV Indir", df.to_csv(index=False).encode('utf-8'),
                           file_name=f"{active_corpus['name']}_folios.csv", mime="text/csv", use_container_width=True)
