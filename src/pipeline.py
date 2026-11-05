"""
Voynich Main Pipeline & Orchestrator
Integrates vision segmentation, unsupervised glyph clustering, slot grammar morphology,
information theory (Shannon & Rényi), historical cipher simulation, and computational philology.
Powered by the high-precision Voynich Graph Engine and ZL Corpus Parser.
"""

import os
import re
import numpy as np
from typing import Dict, Any, List, Optional
from src.vision_segmenter import load_image, binarize_page
from src.glyph_clustering import extract_glyphs_from_word, cluster_glyphs
from src.slot_grammar import parse_token, analyze_corpus_morphology
from src.crypto_entropy import (
    compute_shannon_entropy, compute_index_of_coincidence, analyze_zipf_law,
    sukhotin_vowel_detector, classify_currier_dialect, compute_renyi_entropy,
    compute_multiscale_shannon_entropy, compute_advanced_ic, compute_cross_entropy_and_kl
)
from src.crib_engine import match_folio_cribs
from src.graph_engine import layout_page, align_words_to_tokens, find_matching_folios, DEFAULT_PARAMS
from src.zl_corpus import load_zl, normalize_folio_id
from src.historical_ciphers import (
    CardanGrilleSimulator, HomophonicCipherEngine, PolyalphabeticSolver
)
from src.philology import (
    analyze_hapax_and_mandelbrot, analyze_curio_repetitions, compute_slot_transition_matrix
)


def load_folio_transcription(transcript_path: str, clean_fid: str) -> List[Dict[str, Any]]:
    """
    Extract lines and tokens for a specific folio from ZL transcription.
    """
    zl = load_zl(transcript_path)
    rec = zl.get(clean_fid)
    if not rec:
        return []
    out = []
    for l in rec['lines']:
        out.append({
            'line_number': l['n'],
            'raw_line': l['raw'],
            'clean_text': ' '.join(l['tokens']),
            'tokens': l['tokens']
        })
    return out


def run_folio_analysis(image_path: str, folio_id: str,
                       params: Optional[Dict[str, Any]] = None,
                       transcript_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes full multi-layer analysis on a folio:
    1. Vision: Graph-based scale-adaptive layout segmentation (lines, words, blocks, components)
    2. Transcription & Ground-Truth Alignment (Dynamic Programming)
    3. Morphological Slot Grammar
    4. Information Theory (Shannon & Rényi Entropy, Advanced IoC)
    5. Historical Cipher Simulations (Cardan, Homophonic, Kasiski / Autocorrelation)
    6. Computational Philology (Hapax, Zipf-Mandelbrot, Curio Repetitions)
    7. Sukhotin Vowel Detection
    8. Botanical Crib & Isomorph Matching
    """
    clean_fid = normalize_folio_id(folio_id)
    bgr_img = load_image(image_path)

    # 1. Vision Layer via Graph Engine
    active_params = dict(DEFAULT_PARAMS)
    if params:
        active_params.update(params)

    layout = layout_page(bgr_img, params=active_params)

    # Extract sample glyphs for unsupervised clustering
    extracted_glyphs = []
    bin_img = binarize_page(bgr_img)
    for w in layout['words'][:30]:
        extracted_glyphs.extend(extract_glyphs_from_word(bin_img, w['bbox']))

    clustering_summary = {}
    if extracted_glyphs:
        labels, cluster_indices, _ = cluster_glyphs(extracted_glyphs, n_clusters=min(18, max(2, len(extracted_glyphs)//3)))
        clustering_summary = {
            'glyphs_extracted_count': len(extracted_glyphs),
            'clusters_discovered': len(cluster_indices),
            'top_cluster_sizes': [len(v) for v in sorted(cluster_indices.values(), key=len, reverse=True)[:5]]
        }

    # 2. Transcription Layer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if transcript_path is None:
        transcript_path = os.path.join(base_dir, 'data', 'transcripts', 'ZL3b-n.txt')

    lines_data = load_folio_transcription(transcript_path, clean_fid)
    all_tokens = [tok for l in lines_data for tok in l['tokens']]

    zl_corpus_all = load_zl(transcript_path)
    candidate_folios = []

    # If this folio has no transcription or is custom, match by layout profile
    if not all_tokens:
        candidate_folios = find_matching_folios(layout, zl_corpus_all, top_k=6)

    # Word-to-Token Alignment via DP
    ref_lines_tokens = [l['tokens'] for l in lines_data if l['tokens']]
    alignment = align_words_to_tokens(layout, ref_lines_tokens)

    # 3. Morphology & Slot Grammar Layer
    morph_data = analyze_corpus_morphology(all_tokens)

    # 4. Information Theory & Entropy Layer
    full_text = ' '.join(all_tokens)
    entropy_data = compute_shannon_entropy(full_text)
    multiscale_entropy = compute_multiscale_shannon_entropy(all_tokens)
    renyi_data = compute_renyi_entropy(full_text)
    advanced_ic = compute_advanced_ic(full_text)
    zipf_data = analyze_zipf_law(all_tokens)
    vowels = sukhotin_vowel_detector(full_text, top_vowels=4)
    currier_diag = classify_currier_dialect(all_tokens)

    # 5. Historical Cipher Simulations
    historical_sims = {
        'cardan': CardanGrilleSimulator(k=6).simulate_search(full_text, n_trials=16),
        'kasiski': PolyalphabeticSolver().kasiski_examination(full_text),
        'autocorrelation': PolyalphabeticSolver().autocorrelation_test(full_text),
        'homophonic': HomophonicCipherEngine().analyze_homophonic_clusters(all_tokens)
    }

    # 6. Computational Philology Layer
    philology_data = {
        'hapax_mandelbrot': analyze_hapax_and_mandelbrot(all_tokens),
        'curio_repetitions': analyze_curio_repetitions(all_tokens),
        'slot_transitions': compute_slot_transition_matrix(all_tokens)
    }

    # 7. Crib & Botanical Matching
    crib_data = match_folio_cribs(clean_fid, all_tokens)

    return {
        'folio_id': clean_fid,
        'image_path': image_path,
        'vision': {
            'detected_lines': layout['stats'].get('logical_lines', len(layout['segments'])),
            'detected_words': layout['stats'].get('words', len(layout['words'])),
            'layout_stats': layout['stats'],
            'clustering': clustering_summary
        },
        'graph_layout': layout,
        'alignment': alignment,
        'candidate_folios': candidate_folios,
        'text_corpus': {
            'transcribed_lines': len(lines_data),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'lines': lines_data
        },
        'morphology': morph_data,
        'cryptanalysis': {
            'shannon_h1': entropy_data['h1'],
            'shannon_h2': entropy_data['h2'],
            'multiscale': multiscale_entropy,
            'renyi': renyi_data,
            'index_of_coincidence': advanced_ic['ic'],
            'advanced_ic': advanced_ic,
            'zipf_alpha': zipf_data['alpha'],
            'zipf_r_squared': zipf_data['r_squared'],
            'sukhotin_detected_vowels': vowels,
            'currier_dialect': currier_diag
        },
        'historical_ciphers': historical_sims,
        'philology': philology_data,
        'crib_matching': crib_data
    }
