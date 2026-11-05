"""
Unit and Integration Tests for Voynich System Pipeline
"""

import os
import sys
import pytest
import numpy as np

# Add src to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.vision_segmenter import load_image, binarize_page, segment_lines, process_page
from src.glyph_clustering import extract_glyphs_from_word, cluster_glyphs
from src.slot_grammar import parse_token, analyze_corpus_morphology
from src.crypto_entropy import compute_shannon_entropy, compute_index_of_coincidence, analyze_zipf_law, sukhotin_vowel_detector, classify_currier_dialect
from src.crib_engine import compute_isomorphic_pattern, match_folio_cribs
from src.pipeline import run_folio_analysis


def test_slot_grammar_parsing():
    """Test morphological decomposition of known Voynich canonical words."""
    p1 = parse_token('qokedy')
    assert p1['prefix'] == 'qo'
    assert p1['medial'] == 'k'
    assert p1['core'] == 'e'
    assert p1['suffix'] == 'dy'
    assert p1['is_valid'] is True

    p2 = parse_token('daiin')
    assert p2['prefix'] == 'd'
    assert p2['core'] == 'a'
    assert p2['suffix'] == 'iin'
    assert p2['is_valid'] is True

    p3 = parse_token('otchor')
    assert p3['prefix'] in ['o', 'ot']
    assert p3['medial'] in ['t', 'ch', 'tch']
    assert p3['suffix'] == 'or'
    assert p3['is_valid'] is True

    corpus_stats = analyze_corpus_morphology(['qokedy', 'daiin', 'chedy', 'shedy', 'otchor', 'cthaiin'])
    assert corpus_stats['valid_rate'] >= 0.80


def test_crypto_entropy():
    """Test Shannon entropy, IC, Sukhotin vowel detection, and Currier classifier."""
    sample_voynich = 'daiin qokedy chedy shedy otchor daiin chol chor qokain'
    entropy = compute_shannon_entropy(sample_voynich)
    assert entropy['h1'] > 0.0
    assert entropy['h2'] >= 0.0

    ic = compute_index_of_coincidence(sample_voynich)
    assert 0.0 < ic < 0.2

    tokens = sample_voynich.split()
    zipf = analyze_zipf_law(tokens)
    assert zipf['alpha'] > 0.0

    # Sukhotin vowel detector test
    vowels = sukhotin_vowel_detector(sample_voynich, top_vowels=3)
    assert len(vowels) > 0
    assert any(v in ['o', 'e', 'a', 'i', 'y'] for v in vowels)

    # Currier dialect test (Herbal A markers)
    herbal_tokens = ['daiin', 'chol', 'chor', 'daiin', 'or', 'dal', 'cthor']
    currier_res = classify_currier_dialect(herbal_tokens)
    assert currier_res['dialect'] == 'Currier Language A'


def test_crib_engine():
    """Test isomorphic pattern generation and botanical crib matching."""
    pat_cannabis = compute_isomorphic_pattern('cannabis')
    pat_voynich_repeat = compute_isomorphic_pattern('daiin')
    assert pat_voynich_repeat == '0.1.2.2.3'

    tokens = ['daiin', 'schoal', 'cfhy', 'silene', 'shoyty']
    res = match_folio_cribs('11r', tokens)
    assert res['folio_id'] == '11r'
    assert len(res['botanical_candidates']) >= 1


def test_vision_and_pipeline():
    """End-to-end pipeline test on f11r.png."""
    img_path = os.path.join(base_dir, 'data', 'raw_images', 'f11r.png')
    if not os.path.exists(img_path):
        pytest.skip("f11r.png not found, skipping vision test")

    res = run_folio_analysis(img_path, '11r')
    assert res['folio_id'] == '11r'
    assert res['vision']['detected_lines'] > 0
    assert res['vision']['detected_words'] > 0
    assert res['text_corpus']['transcribed_lines'] > 0
    assert res['morphology']['valid_rate'] >= 0.60
    assert res['cryptanalysis']['currier_dialect']['dialect'] == 'Currier Language A'
