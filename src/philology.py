"""
Voynich Computational Philology & Linguistic Morphology
-------------------------------------------------------
Implements deep statistical philology:
1. Hapax Legomena, Dis Legomena, and Zipf-Mandelbrot Law parameter estimation.
2. Curio Repetitions (Immediate Word-Level Echoes and Positional Affinities).
3. Currier Language A vs Language B Dialect Discriminator.
4. Slot-and-Filler Morphological Transition Probability Matrix.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.slot_grammar import parse_token, PREFIXES, GALLOWS_BENCHES, CORES, SUFFIXES


# -----------------------------------------------------------------------------
# 1. HAPAX LEGOMENA & ZIPF-MANDELBROT ANALYZER
# -----------------------------------------------------------------------------
def analyze_hapax_and_mandelbrot(tokens: List[str]) -> Dict[str, Any]:
    """
    Computes vocabulary distribution parameters:
    - Hapax Legomena (f = 1), Dis Legomena (f = 2), Tris Legomena (f = 3)
    - Mandelbrot-Zipf parameter estimation: f(r) = C / (r + beta)^alpha
    - Heaps' law vocabulary growth exponent
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
    n = len(clean_tokens)
    if n == 0:
        return {
            'total_tokens': 0, 'vocab_size': 0,
            'hapax_count': 0, 'hapax_ratio': 0.0,
            'dis_legomena_count': 0, 'mandelbrot_alpha': 0.0,
            'mandelbrot_beta': 0.0, 'heaps_beta': 0.0
        }

    counts = Counter(clean_tokens)
    v = len(counts)

    hapax = [w for w, cnt in counts.items() if cnt == 1]
    dis = [w for w, cnt in counts.items() if cnt == 2]
    tris = [w for w, cnt in counts.items() if cnt == 3]

    # Ranks and Frequencies
    sorted_counts = counts.most_common()
    ranks = np.arange(1, v + 1, dtype=float)
    freqs = np.array([cnt for _, cnt in sorted_counts], dtype=float)

    # Fit Mandelbrot-Zipf: log(freq) = log(C) - alpha * log(rank + beta)
    # Search optimal beta in [0, 20]
    best_beta = 0.0
    best_alpha = 1.0
    best_r2 = -1.0
    for beta_cand in np.linspace(0.0, 15.0, 31):
        log_x = np.log10(ranks + beta_cand)
        log_y = np.log10(freqs)
        if len(ranks) > 2:
            p = np.polyfit(log_x, log_y, 1)
            y_pred = np.polyval(p, log_x)
            r2 = 1.0 - (np.sum((log_y - y_pred)**2) / max(np.sum((log_y - np.mean(log_y))**2), 1e-9))
            if r2 > best_r2:
                best_r2 = r2
                best_beta = beta_cand
                best_alpha = abs(p[0])

    # Heaps' Law: V(N) = K * N^beta across progressive prefixes
    heaps_exp = 0.60
    if n >= 50:
        step = max(5, n // 10)
        sub_n = []
        sub_v = []
        seen = set()
        for i, t in enumerate(clean_tokens):
            seen.add(t)
            if (i + 1) % step == 0:
                sub_n.append(i + 1)
                sub_v.append(len(seen))
        if len(sub_n) >= 3:
            p_heaps = np.polyfit(np.log10(sub_n), np.log10(sub_v), 1)
            heaps_exp = float(np.clip(p_heaps[0], 0.1, 1.0))

    return {
        'total_tokens': n,
        'vocabulary_size': v,
        'hapax_count': len(hapax),
        'hapax_ratio_token': round(len(hapax) / n, 4),
        'hapax_ratio_vocab': round(len(hapax) / v, 4),
        'dis_legomena_count': len(dis),
        'tris_legomena_count': len(tris),
        'mandelbrot_alpha': round(float(best_alpha), 4),
        'mandelbrot_beta': round(float(best_beta), 2),
        'mandelbrot_r2': round(float(max(0.0, best_r2)), 4),
        'heaps_growth_exponent': round(heaps_exp, 3),
        'sample_hapax': hapax[:12],
        'sample_dis': dis[:8]
    }


# -----------------------------------------------------------------------------
# 2. CURIO REPETITIONS & ECHO PHENOMENON ANALYZER
# -----------------------------------------------------------------------------
def analyze_curio_repetitions(tokens: List[str]) -> Dict[str, Any]:
    """
    Analyzes the remarkable Voynich phenomenon of immediate word duplications
    (e.g., 'qokedy qokedy' or 'chol chol').
    Natural medieval texts average 0.1% - 0.3% immediate repetitions.
    Voynich manuscript exhibits 1.5% - 2.8% immediate repetitions.
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
    n = len(clean_tokens)
    if n < 4:
        return {
            'total_tokens': n, 'immediate_repeats_count': 0,
            'duplication_rate': 0.0, 'near_repeats_count': 0,
            'repeated_pairs': [], 'anomaly_factor': 1.0
        }

    immediate_pairs = []
    near_pairs = []

    for i in range(n - 1):
        if clean_tokens[i] == clean_tokens[i+1]:
            immediate_pairs.append({
                'token': clean_tokens[i],
                'position': i,
                'distance': 1
            })

    for i in range(n - 2):
        if clean_tokens[i] == clean_tokens[i+2] and clean_tokens[i] != clean_tokens[i+1]:
            near_pairs.append({
                'token': clean_tokens[i],
                'intermediate': clean_tokens[i+1],
                'position': i,
                'distance': 2
            })

    imm_count = len(immediate_pairs)
    imm_rate = imm_count / (n - 1)
    near_count = len(near_pairs)

    # Anomaly factor compared to typical Latin/Medieval text rate (approx 0.002)
    natural_rate = 0.002
    anomaly_factor = round(imm_rate / max(natural_rate, 1e-6), 1)

    top_repeated_tokens = Counter(p['token'] for p in immediate_pairs).most_common(8)

    return {
        'total_tokens': n,
        'immediate_repeats_count': imm_count,
        'duplication_rate_percent': round(imm_rate * 100, 2),
        'near_repeats_count': near_count,
        'near_repeats_rate_percent': round(near_count / max(n - 2, 1) * 100, 2),
        'anomaly_factor_vs_natural': anomaly_factor,
        'top_duplicated_tokens': [{'token': t, 'count': c} for t, c in top_repeated_tokens],
        'sample_immediate_pairs': immediate_pairs[:12]
    }


# -----------------------------------------------------------------------------
# 3. CURRIER LANGUAGE A VS LANGUAGE B DISCRIMINATOR
# -----------------------------------------------------------------------------
def analyze_currier_dialect_overlap(tokens_a: List[str], tokens_b: List[str]) -> Dict[str, Any]:
    """
    Computes vocabulary intersection, mutual exclusivity, and distinguishing tokens
    between Currier Language A and Currier Language B.
    """
    set_a = set(t.lower().strip() for t in tokens_a if t.strip())
    set_b = set(t.lower().strip() for t in tokens_b if t.strip())

    intersection = set_a.intersection(set_b)
    exclusive_a = set_a.difference(set_b)
    exclusive_b = set_b.difference(set_a)

    jaccard = len(intersection) / max(len(set_a.union(set_b)), 1)

    # Calculate token frequencies in each corpus
    counts_a = Counter(t.lower().strip() for t in tokens_a if t.strip())
    counts_b = Counter(t.lower().strip() for t in tokens_b if t.strip())

    tot_a = sum(counts_a.values())
    tot_b = sum(counts_b.values())

    # Log-odds ratio for diagnostic markers
    diagnostic_a = []
    for w in exclusive_a:
        if counts_a[w] >= 3:
            diagnostic_a.append({'token': w, 'freq_a': counts_a[w], 'dialect': 'A'})
    diagnostic_a.sort(key=lambda x: -x['freq_a'])

    diagnostic_b = []
    for w in exclusive_b:
        if counts_b[w] >= 3:
            diagnostic_b.append({'token': w, 'freq_b': counts_b[w], 'dialect': 'B'})
    diagnostic_b.sort(key=lambda x: -x['freq_b'])

    return {
        'vocab_size_a': len(set_a),
        'vocab_size_b': len(set_b),
        'shared_vocab_size': len(intersection),
        'jaccard_similarity': round(jaccard, 3),
        'exclusive_a_count': len(exclusive_a),
        'exclusive_b_count': len(exclusive_b),
        'top_markers_exclusive_a': diagnostic_a[:10],
        'top_markers_exclusive_b': diagnostic_b[:10]
    }


# -----------------------------------------------------------------------------
# 4. SLOT-AND-FILLER MORPHOLOGICAL TRANSITION MATRIX
# -----------------------------------------------------------------------------
def compute_slot_transition_matrix(tokens: List[str]) -> Dict[str, Any]:
    """
    Builds the Markov transition probabilities between morphological slots:
    START -> Prefix -> Medial/Gallows -> Core -> Suffix -> END
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
    if not clean_tokens:
        return {'valid_ratio': 0.0, 'transitions': {}, 'prefix_to_core': {}}

    prefix_to_core = defaultdict(Counter)
    core_to_suffix = defaultdict(Counter)
    prefix_to_medial = defaultdict(Counter)

    parsed_count = 0
    valid_count = 0

    for tok in clean_tokens:
        p = parse_token(tok)
        parsed_count += 1
        if p['is_valid']:
            valid_count += 1
            pref = p['prefix'] or '<NONE>'
            med = p['medial'] or '<NONE>'
            core = p['core'] or '<NONE>'
            suf = p['suffix'] or '<NONE>'

            prefix_to_medial[pref][med] += 1
            prefix_to_core[pref][core] += 1
            core_to_suffix[core][suf] += 1

    # Format top transitions
    top_p_c = []
    for pref, core_counts in prefix_to_core.items():
        tot = sum(core_counts.values())
        top_core, c_num = core_counts.most_common(1)[0]
        prob = c_num / tot
        top_p_c.append({
            'prefix': pref,
            'most_likely_core': top_core,
            'probability': round(prob, 2),
            'count': tot
        })
    top_p_c.sort(key=lambda x: -x['count'])

    return {
        'valid_rate': round(valid_count / max(parsed_count, 1), 3),
        'prefix_to_core_transitions': top_p_c[:12]
    }
