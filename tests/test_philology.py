"""
Unit Tests for Computational Philology & Extended Information Theory
"""

import os
import sys
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.philology import (
    analyze_hapax_and_mandelbrot, analyze_curio_repetitions,
    analyze_currier_dialect_overlap, compute_slot_transition_matrix
)
from src.crypto_entropy import (
    compute_renyi_entropy, compute_multiscale_shannon_entropy,
    compute_advanced_ic, compute_cross_entropy_and_kl
)


def test_hapax_and_mandelbrot():
    tokens = ["daiin", "daiin", "qokedy", "chedy", "unique_word", "shedy", "another_hapax"]
    res = analyze_hapax_and_mandelbrot(tokens)
    assert res['total_tokens'] == 7
    assert res['vocabulary_size'] == 6
    assert res['hapax_count'] >= 2
    assert res['mandelbrot_alpha'] > 0.0


def test_curio_repetitions():
    tokens = ["qokedy", "qokedy", "chedy", "daiin", "daiin", "daiin", "chol"]
    res = analyze_curio_repetitions(tokens)
    assert res['immediate_repeats_count'] >= 3
    assert res['duplication_rate_percent'] > 0.0
    assert len(res['top_duplicated_tokens']) >= 1


def test_currier_dialect_overlap():
    tokens_a = ["daiin", "chol", "chor", "or", "al", "daiin", "dal"] * 4
    tokens_b = ["chedy", "shedy", "qokedy", "qokain", "chedy"] * 4
    res = analyze_currier_dialect_overlap(tokens_a, tokens_b)
    assert res['vocab_size_a'] > 0
    assert res['vocab_size_b'] > 0
    assert res['jaccard_similarity'] <= 0.5


def test_slot_transition_matrix():
    tokens = ["qokedy", "daiin", "otchor", "chedy", "shedy"] * 3
    matrix = compute_slot_transition_matrix(tokens)
    assert matrix['valid_rate'] >= 0.70
    assert len(matrix['prefix_to_core_transitions']) >= 1


def test_renyi_entropy_spectrum():
    text = "daiin qokedy chedy shedy otchor chol"
    renyi = compute_renyi_entropy(text)
    assert 'H_0 (Hartley)' in renyi
    assert 'H_1 (Shannon)' in renyi
    assert 'H_2 (Carpisma)' in renyi
    assert 'H_inf (Min-Entropi)' in renyi
    # By Rényi properties, H_0 >= H_1 >= H_2 >= H_inf
    assert renyi['H_0 (Hartley)'] >= renyi['H_1 (Shannon)'] >= renyi['H_2 (Carpisma)'] >= renyi['H_inf (Min-Entropi)']


def test_multiscale_shannon_and_advanced_ic():
    tokens = ["daiin", "qokedy", "chedy", "shedy", "otchor"] * 5
    ms = compute_multiscale_shannon_entropy(tokens)
    assert ms['h1'] > 0.0
    assert ms['h2'] >= 0.0
    assert ms['h3'] >= 0.0
    assert ms['h_word'] > 0.0

    ic = compute_advanced_ic(" ".join(tokens))
    assert 'ic' in ic
    assert 'normalized_ic' in ic
    assert 'verdict' in ic

    ce = compute_cross_entropy_and_kl(tokens, target_lang='latin')
    assert ce['cross_entropy'] > 0.0
    assert ce['kl_divergence'] >= 0.0
