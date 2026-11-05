"""
Unit tests for e2Canvas Sandbox backend and component architecture.
"""

import pytest
import numpy as np
from src.e2canvas_backend import (
    _levenshtein, compute_dictionary_match, compute_bigram_coherence,
    analyze_canvas_cluster
)
from web.components.editor_iiif import (
    render_transcription_editor, render_openseadragon_iiif_viewer
)
from web.components.e2canvas import render_e2canvas


def test_levenshtein_distance():
    assert _levenshtein("folium", "folium") == 0
    assert _levenshtein("folium", "folia") == 2
    assert _levenshtein("radix", "radice") == 2
    assert _levenshtein("", "test") == 4


def test_compute_dictionary_match():
    # Exact hit
    score, candidate = compute_dictionary_match("folium")
    assert score == 100.0
    assert candidate == "folium"

    # Close hit
    score_close, cand_close = compute_dictionary_match("radice")
    assert score_close == 100.0  # radice is in Medieval Italian lexicon!

    score_approx, cand_approx = compute_dictionary_match("folius")
    assert score_approx > 75.0
    assert cand_approx in ("folium", "foliis")

    # Empty string
    score_empty, cand_empty = compute_dictionary_match("")
    assert score_empty == 0.0
    assert cand_empty is None


def test_compute_bigram_coherence():
    # Canonical Voynich words with standard bigrams (da, ai, in, qo, ok, ed, dy, ch, etc.)
    voynich_tokens = ["daiin", "qokedy", "chedy", "shedy"]
    coherence = compute_bigram_coherence(voynich_tokens)
    assert coherence >= 70.0

    # Non-conforming random consonants
    gibberish = ["zxq", "bjk", "wvv"]
    coherence_gibberish = compute_bigram_coherence(gibberish)
    assert coherence_gibberish == 0.0

    # Empty list
    assert compute_bigram_coherence([]) == 0.0


def test_analyze_canvas_cluster():
    nodes = [
        {"id": "node_1", "type": "crop", "eva": "daiin", "label": "daiin"},
        {"id": "node_2", "type": "word", "eva": "qokedy", "label": "qokedy"},
        {"id": "node_3", "type": "trans", "translation": "folium", "label": "folium"}
    ]
    res = analyze_canvas_cluster(nodes)

    assert res["node_count"] == 3
    assert res["shannon_h1"] > 0.0
    assert res["shannon_h2"] >= 0.0
    assert res["bigram_coherence"] > 60.0
    assert res["dictionary_score"] >= 80.0  # "folium" is in lexicon
    assert len(res["morphology"]) >= 2
    assert "Shannon H:" in res["summary_line"]
    assert "Bigram Tutarlılığı:" in res["summary_line"]


def test_component_functions_run_without_errors():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [{"bbox": (10, 10, 30, 20), "conf": 0.95, "eva": "daiin"}]

    # Smoke test rendering functions (do not raise exceptions)
    render_transcription_editor("daiin qokedy", mode="EVA", height=200)
    render_openseadragon_iiif_viewer(dummy_img, boxes, height=300, folio_id="f11r")
    render_e2canvas(
        folio_id="f11r",
        crops=[{"id": "c1", "eva": "daiin", "img_b64": "", "conf": 0.9}],
        words=[{"eva": "qokedy", "count": 2, "line": 1, "slot": "core"}],
        hypotheses=[{"eva": "daiin", "translation": "folium", "lang": "Latince"}],
        height=400
    )
