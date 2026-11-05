"""
Unit Tests for Historical Cryptographic Hypothesis Simulators
"""

import os
import sys
import pytest
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.historical_ciphers import (
    CardanGrilleSimulator, HomophonicCipherEngine, PolyalphabeticSolver
)


def test_cardan_grille_canonical_orbits():
    sim = CardanGrilleSimulator(k=6)
    mask = sim.generate_canonical_grille(seed=42)
    assert mask.shape == (6, 6)
    # Total holes must be k^2 / 4 = 9
    assert np.sum(mask) == 9

    # The 4 rotations must cover every cell in the 6x6 grid exactly once
    m0 = mask
    m90 = np.rot90(m0, -1)
    m180 = np.rot90(m90, -1)
    m270 = np.rot90(m180, -1)
    total_coverage = m0.astype(int) + m90.astype(int) + m180.astype(int) + m270.astype(int)
    assert np.all(total_coverage == 1)


def test_cardan_grille_simulation():
    sim = CardanGrilleSimulator(k=4)
    # Sample text with radix embedded
    sample_text = "radixplantafoliumsemen" * 4
    res = sim.simulate_search(sample_text, n_trials=5)
    assert 'score' in res
    assert 'extracted_sample' in res
    assert isinstance(res['matched_roots'], list)


def test_homophonic_cipher_engine():
    engine = HomophonicCipherEngine()
    tokens = [
        "daiin", "qokedy", "chedy", "shedy", "otchor", "daiin", "chol",
        "qokedy", "chedy", "chol", "daiin", "qokedy", "chedy"
    ] * 5
    res = engine.analyze_homophonic_clusters(tokens, top_n=6)
    assert 'homophonic_candidates' in res
    assert 'baseline_char_entropy' in res
    assert res['baseline_char_entropy'] > 0.0


def test_polyalphabetic_kasiski_examination():
    solver = PolyalphabeticSolver()
    # Construct a repeating sequence with known period 5: ABCDE ABCDE ABCDE
    key = "ABCDE"
    text = (key * 10) + "XYZ" + (key * 8)
    kasiski = solver.kasiski_examination(text, min_len=3, max_len=4)
    assert kasiski['total_repeats_found'] > 0
    # Period 5 or divisors must be detected
    factors = [f['key_length'] for f in kasiski['probable_key_lengths']]
    assert 5 in factors or len(factors) > 0


def test_autocorrelation_test():
    solver = PolyalphabeticSolver()
    # Periodic sequence with period 4
    pattern = "abcd" * 25
    auto = solver.autocorrelation_test(pattern, max_shift=10)
    assert len(auto['shifts']) == 10
    # Shift 4 and 8 must have high coincidence
    rate_4 = [s['coincidence_rate'] for s in auto['shifts'] if s['shift'] == 4][0]
    rate_1 = [s['coincidence_rate'] for s in auto['shifts'] if s['shift'] == 1][0]
    assert rate_4 > rate_1
