"""
Unit Tests for Cryptologist Studio & Graph Alignment Engine
"""

import os
import sys
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.crypto_studio import (
    SubstitutionCipher,
    PageAnnotationManager,
    GlobalLexicon,
    CIPHER_PRESETS,
    MEDIEVAL_BOTANICAL_LEXICON
)
from src.zl_corpus import load_zl, folio_tokens
from src.graph_engine import layout_page, align_words_to_tokens, word_crop
import cv2
import numpy as np


def test_substitution_cipher():
    cipher = SubstitutionCipher()
    assert cipher.decipher_token("daiin") == "daiin"

    # Custom mapping: d->t, a->e, i->l, n->m => "tellm"
    cipher.set_mapping('d', 't')
    cipher.set_mapping('a', 'e')
    cipher.set_mapping('i', 'l')
    cipher.set_mapping('n', 'm')
    assert cipher.decipher_token("daiin") == "tellm"

    # Evaluation with a known word from medieval botanical lexicon
    cipher_herb = SubstitutionCipher({'r': 'r', 'a': 'a', 'd': 'd', 'i': 'i', 'x': 'x'})
    eval_res = cipher_herb.evaluate_decipherment(["radix", "folium", "aqua", "flos"])
    assert eval_res['total_words'] == 4
    assert eval_res['lexicon_hits'] >= 3
    assert eval_res['hit_rate'] >= 75.0
    assert eval_res['plausibility_score'] > 50.0


def test_cipher_presets():
    assert "Latince Fonetik Hipotezi" in CIPHER_PRESETS
    lat = SubstitutionCipher(CIPHER_PRESETS["Latince Fonetik Hipotezi"])
    dec = lat.decipher_token("qokedy")
    assert dec != ""
    assert isinstance(dec, str)


def test_annotation_manager(tmp_path):
    mgr = PageAnnotationManager("11r")
    mgr.set_word_annotation(
        word_idx=0,
        reading="daiin",
        meaning="yaprak (folium)",
        pos="İsim",
        notes="Tekrar eden yaprak formülü",
        confidence=4
    )
    ann = mgr.get_word_annotation(0)
    assert ann is not None
    assert ann['reading'] == "daiin"
    assert ann['meaning'] == "yaprak (folium)"
    assert ann['confidence'] == 4


def test_global_lexicon(tmp_path):
    test_file = str(tmp_path / "test_lexicon.json")
    lex = GlobalLexicon(test_file)
    lex.add_entry("daiin", "yaprak", "İsim", source_folio="f11r", notes="İlk kelime")
    entry = lex.lookup("daiin")
    assert entry is not None
    assert entry['meaning'] == "yaprak"
    assert "f11r" in entry['folios']

    # Update entry
    lex.add_entry("daiin", "yaprak / folium", "İsim", source_folio="f16r")
    updated = lex.lookup("daiin")
    assert "f16r" in updated['folios']
    assert updated['meaning'] == "yaprak / folium"


def test_graph_engine_layout_and_alignment():
    # Create synthetic test page image with 2 lines of text
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    # Line 1: two words
    cv2.putText(img, "daiin qokedy", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 30, 20), 2)
    # Line 2: three words
    cv2.putText(img, "chedy shedy chol", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 30, 20), 2)

    layout = layout_page(img)
    assert layout['stats']['components'] > 0
    assert len(layout['words']) >= 2

    # Test word crop
    wc = word_crop(img, layout['words'][0])
    assert wc.shape[0] > 0 and wc.shape[1] > 0

    # Test alignment to synthetic transcription
    zl_lines = [["daiin", "qokedy"], ["chedy", "shedy", "chol"]]
    alignment = align_words_to_tokens(layout, zl_lines)
    assert 'word_tokens' in alignment
    assert 'line_pairs' in alignment


def test_categorized_cipher_presets():
    from src.crypto_studio import CATEGORIZED_CIPHER_PRESETS
    assert "Antik Diller & Alfabeler" in CATEGORIZED_CIPHER_PRESETS
    assert "Hiyeroglifler & Çivi Yazısı" in CATEGORIZED_CIPHER_PRESETS
    assert "En Yaygın 10 Modern Dil" in CATEGORIZED_CIPHER_PRESETS

    # Check that Greek and Hebrew exist
    antik = CATEGORIZED_CIPHER_PRESETS["Antik Diller & Alfabeler"]
    assert any("Grek" in k for k in antik.keys())
    assert any("İbrani" in k or "Ibrani" in k for k in antik.keys())

    # Check top modern languages include Turkish, Chinese, Russian, English
    modern = CATEGORIZED_CIPHER_PRESETS["En Yaygın 10 Modern Dil"]
    assert any("Türkçe" in k or "Turkce" in k for k in modern.keys())
    assert any("İngilizce" in k or "Ingilizce" in k for k in modern.keys())


def test_grammar_profiles_and_comparison():
    from src.slot_grammar import GRAMMAR_PROFILES, compare_grammars_on_tokens
    assert any("Klasik Voynich" in k for k in GRAMMAR_PROFILES)
    assert any("Currier Dil A" in k for k in GRAMMAR_PROFILES)
    assert any("Currier Dil B" in k for k in GRAMMAR_PROFILES)
    assert any("Latin" in k for k in GRAMMAR_PROFILES)

    sample_tokens = ["daiin", "qokedy", "chedy", "shedy", "chol", "otchor", "cthaiin"]
    res = compare_grammars_on_tokens(sample_tokens, list(GRAMMAR_PROFILES.keys()))
    assert len(res['results']) == len(GRAMMAR_PROFILES)
    assert res['best_grammar'] != ""
    assert res['results'][0]['valid_tokens'] > 0


def test_concordance_and_bridge_context():
    from src.e2canvas_backend import find_token_corpus_concordance, analyze_bridge_relationship
    conc = find_token_corpus_concordance("daiin", max_matches=10)
    assert conc['token'] == "daiin"
    assert conc['total_occurrences'] > 0
    assert conc['unique_folios_count'] > 0
    assert len(conc['folios']) > 0

    node1 = {'eva': 'daiin', 'type': 'word'}
    node2 = {'eva': 'qokedy', 'type': 'word'}
    bridge = analyze_bridge_relationship(node1, node2)
    assert 'summary' in bridge
    assert 'collocation_frequency' in bridge

