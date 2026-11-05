"""
Voynich Information Theory & Cryptanalysis
Computes:
- Shannon character, n-gram, and word-level entropy (H1, H2, H3, H_word)
- Rényi generalized entropy spectrum (H_alpha for alpha in {0, 0.5, 1, 2, inf})
- Index of Coincidence (IoC & Normalized IC) with mono vs polyalphabetic hypothesis testing
- Cross-entropy and Kullback-Leibler divergence against reference historical languages
- Sukhotin's vowel detection algorithm
- Zipf's Law and Currier Language A vs B classification
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Reference character unigram distributions for historical languages (frequencies normalized to 1.0)
LANGUAGE_CHAR_FREQS = {
    'latin': {
        'e': 0.118, 'i': 0.106, 't': 0.088, 'a': 0.082, 'u': 0.076,
        's': 0.074, 'r': 0.068, 'n': 0.064, 'm': 0.059, 'o': 0.054,
        'c': 0.041, 'd': 0.038, 'l': 0.035, 'p': 0.031, 'b': 0.016,
        'v': 0.015, 'g': 0.014, 'f': 0.010, 'h': 0.009, 'q': 0.008, 'x': 0.005
    },
    'italian': {
        'e': 0.118, 'a': 0.117, 'i': 0.113, 'o': 0.098, 'n': 0.069,
        'r': 0.064, 't': 0.056, 'l': 0.065, 's': 0.050, 'c': 0.045,
        'd': 0.037, 'u': 0.030, 'm': 0.025, 'p': 0.030, 'v': 0.021,
        'b': 0.009, 'f': 0.010, 'g': 0.016, 'h': 0.015, 'z': 0.012
    },
    'greek': {
        'α': 0.115, 'ε': 0.098, 'ο': 0.095, 'ι': 0.085, 'τ': 0.079,
        'ν': 0.075, 'σ': 0.069, 'ρ': 0.055, 'η': 0.049, 'υ': 0.044,
        'π': 0.039, 'κ': 0.038, 'μ': 0.037, 'λ': 0.034, 'ω': 0.025,
        'δ': 0.022, 'γ': 0.019, 'θ': 0.015, 'χ': 0.012, 'φ': 0.009
    },
    'hebrew': {
        'י': 0.108, 'ו': 0.103, 'ה': 0.088, 'ל': 0.074, 'מ': 0.068,
        'ב': 0.058, 'ר': 0.056, 'ת': 0.055, 'א': 0.053, 'ש': 0.045,
        'ע': 0.039, 'נ': 0.038, 'כ': 0.036, 'ד': 0.031, 'ח': 0.026,
        'ק': 0.024, 'פ': 0.021, 'ס': 0.018, 'צ': 0.014, 'ז': 0.010, 'ט': 0.009
    }
}

# Cross-linguistic standard benchmark Index of Coincidence values
LANGUAGE_IOC_BENCHMARKS = {
    'Voynich (Ortalama)': 0.0415,
    'Latince (Klasik)': 0.0700,
    'Erken İtalyanca': 0.0738,
    'İngilizce': 0.0667,
    'Almanca': 0.0762,
    'Fransızca': 0.0778,
    'Grekçe': 0.0660,
    'İbranice': 0.0780,
    'Rastgele 20 Karakter (Tekdüze)': 0.0500,
    'Rastgele 26 Karakter (Tekdüze)': 0.0385
}


def compute_shannon_entropy(text: str) -> Dict[str, float]:
    """
    Computes 1st order character entropy H1 and 2nd order conditional entropy H2.
    """
    chars = [c for c in text.lower() if c.isalnum()]
    n = len(chars)
    if n < 2:
        return {'h1': 0.0, 'h2': 0.0, 'char_count': n, 'unique_chars': n}

    # H1: Unigram entropy
    freq1 = Counter(chars)
    h1 = -sum((cnt / n) * math.log2(cnt / n) for cnt in freq1.values())

    # H(X, Y): Joint bigram entropy
    bigrams = [chars[i] + chars[i+1] for i in range(n - 1)]
    freq2 = Counter(bigrams)
    n2 = len(bigrams)
    h_joint = -sum((cnt / n2) * math.log2(cnt / n2) for cnt in freq2.values())

    # Conditional entropy H2 = H(X, Y) - H(X)
    h2 = max(0.0, h_joint - h1)

    return {
        'h1': round(h1, 4),
        'h2': round(h2, 4),
        'char_count': n,
        'unique_chars': len(freq1)
    }


def compute_multiscale_shannon_entropy(tokens: List[str]) -> Dict[str, Any]:
    """
    Computes multi-scale information density:
    - Character unigram entropy H1
    - Character conditional bigram entropy H2
    - Character conditional trigram entropy H3
    - Word-level entropy H_word
    - Natural language comparisons
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
    if not clean_tokens:
        return {
            'h1': 0.0, 'h2': 0.0, 'h3': 0.0, 'h_word': 0.0,
            'token_count': 0, 'char_count': 0
        }

    full_text = " ".join(clean_tokens)
    chars = [c for c in full_text if c.isalnum()]
    n_chars = len(chars)

    # 1. H1 (Unigram)
    c1 = Counter(chars)
    h1 = -sum((cnt / n_chars) * math.log2(cnt / n_chars) for cnt in c1.values()) if n_chars > 0 else 0.0

    # 2. H2 (Conditional Bigram)
    h2 = 0.0
    if n_chars > 1:
        c2 = Counter(zip(chars[:-1], chars[1:]))
        tot2 = len(chars) - 1
        h2 = -sum((cnt / tot2) * math.log2(cnt / c1[p[0]]) for p, cnt in c2.items())
        h2 = max(0.0, h2)

    # 3. H3 (Conditional Trigram)
    h3 = 0.0
    if n_chars > 2:
        c3 = Counter(zip(chars[:-2], chars[1:-1], chars[2:]))
        tot3 = len(chars) - 2
        bigram_counts = Counter(zip(chars[:-1], chars[1:]))
        h3 = -sum((cnt / tot3) * math.log2(cnt / bigram_counts[(t[0], t[1])]) for t, cnt in c3.items())
        h3 = max(0.0, h3)

    # 4. Word-level Entropy H_word
    w_counts = Counter(clean_tokens)
    n_words = len(clean_tokens)
    h_word = -sum((cnt / n_words) * math.log2(cnt / n_words) for cnt in w_counts.values()) if n_words > 0 else 0.0

    return {
        'h1': round(h1, 4),
        'h2': round(h2, 4),
        'h3': round(h3, 4),
        'h_word': round(h_word, 4),
        'token_count': n_words,
        'char_count': n_chars,
        'vocabulary_size': len(w_counts),
        'comparisons': {
            'Voynich Metni': {'H1': round(h1, 2), 'H2': round(h2, 2), 'H3': round(h3, 2)},
            'Klasik Latince': {'H1': 4.12, 'H2': 3.75, 'H3': 3.10},
            'Erken İtalyanca': {'H1': 4.18, 'H2': 3.90, 'H3': 3.25},
            'Orta İngilizce': {'H1': 4.35, 'H2': 3.80, 'H3': 3.15},
            'Rastgele Tekdüze': {'H1': 4.70, 'H2': 4.70, 'H3': 4.70}
        }
    }


def compute_renyi_entropy(text: str, alphas: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Computes Rényi generalized entropy spectrum:
      H_alpha(X) = 1 / (1 - alpha) * log2( sum( p_i^alpha ) )
    Special cases:
      alpha = 0: Hartley / log2(|X|)
      alpha = 1: Shannon entropy
      alpha = 2: Collision entropy
      alpha = inf: Min-entropy = -log2( max(p_i) )
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0, float('inf')]

    chars = [c for c in text.lower() if c.isalnum()]
    n = len(chars)
    if n == 0:
        return {f"H_{a}": 0.0 for a in alphas}

    counts = Counter(chars)
    probs = np.array([cnt / n for cnt in counts.values()], dtype=np.float64)

    results = {}
    for a in alphas:
        if a == 0.0:
            # Hartley entropy
            val = math.log2(len(probs))
            results['H_0 (Hartley)'] = round(val, 4)
        elif a == 1.0:
            # Shannon entropy limit
            val = -float(np.sum(probs * np.log2(probs)))
            results['H_1 (Shannon)'] = round(val, 4)
        elif math.isinf(a):
            # Min-entropy
            val = -math.log2(float(np.max(probs)))
            results['H_inf (Min-Entropi)'] = round(val, 4)
        else:
            sum_p_alpha = float(np.sum(probs ** a))
            val = (1.0 / (1.0 - a)) * math.log2(max(sum_p_alpha, 1e-12))
            key = f"H_{a}" if a != 2.0 else "H_2 (Carpisma)"
            results[key] = round(val, 4)

    return results


def compute_index_of_coincidence(text: str) -> float:
    """
    Computes the standard Index of Coincidence (IoC).
    """
    chars = [c for c in text.lower() if c.isalnum()]
    n = len(chars)
    if n <= 1:
        return 0.0

    counts = Counter(chars)
    numerator = sum(f * (f - 1) for f in counts.values())
    denominator = n * (n - 1)
    ic = numerator / denominator
    return round(ic, 5)


def compute_advanced_ic(text: str) -> Dict[str, Any]:
    """
    Computes standard IC, Normalized IC (NIC), and provides cipher hypothesis classification:
    - Monoalphabetic threshold: IC > 0.060
    - Polyalphabetic periodic threshold: 0.045 <= IC <= 0.060
    - Polyalphabetic random / non-language threshold: IC < 0.045
    """
    chars = [c for c in text.lower() if c.isalnum()]
    n = len(chars)
    if n <= 1:
        return {
            'ic': 0.0, 'normalized_ic': 0.0, 'alphabet_size': 0,
            'verdict': 'Yetersiz Veri', 'benchmarks': LANGUAGE_IOC_BENCHMARKS
        }

    counts = Counter(chars)
    c_size = len(counts)
    numerator = sum(f * (f - 1) for f in counts.values())
    denominator = n * (n - 1)
    ic = numerator / denominator if denominator > 0 else 0.0
    nic = ic * c_size

    # Hypothesis interpretation
    if ic >= 0.062:
        verdict = 'Monoalfabetik / Dogal Dil Karakteristigi'
        hypo = 'Metin tek alfabeli ikame veya acik metin dagilimi sergiliyor.'
    elif 0.048 <= ic < 0.062:
        verdict = 'Karisik / Kisa Periyotlu Polialfabetik'
        hypo = 'Kisa anahtarli Vigenere veya hecesel kodlama olabilir.'
    else:
        verdict = 'Cok Alfabeli (Polialfabetik) / Hece-Glif Duzeyi'
        hypo = 'Yuksek homofonik dagilim veya genis glif alfabesi belirtisi.'

    return {
        'ic': round(ic, 5),
        'normalized_ic': round(nic, 4),
        'alphabet_size': c_size,
        'char_count': n,
        'verdict': verdict,
        'hypothesis_note': hypo,
        'benchmarks': LANGUAGE_IOC_BENCHMARKS
    }


def compute_cross_entropy_and_kl(tokens: List[str], target_lang: str = 'latin') -> Dict[str, float]:
    """
    Computes Cross-Entropy H(P, Q) and Kullback-Leibler Divergence D_KL(P || Q)
    between the observed text token character distribution (P) and reference historical language distribution (Q).
    """
    clean_text = "".join(t.lower() for t in tokens if t)
    chars = [c for c in clean_text if c.isalpha()]
    n = len(chars)
    if n < 10 or target_lang.lower() not in LANGUAGE_CHAR_FREQS:
        return {'cross_entropy': 0.0, 'kl_divergence': 0.0, 'sample_chars': n}

    ref_q = LANGUAGE_CHAR_FREQS[target_lang.lower()]
    # Observed P
    counts_p = Counter(chars)
    
    # Smooth over combined alphabet
    all_keys = set(ref_q.keys()).union(set(counts_p.keys()))
    smooth_p = {k: (counts_p.get(k, 0) + 1e-4) / (n + 1e-4 * len(all_keys)) for k in all_keys}
    tot_q = sum(ref_q.values())
    smooth_q = {k: (ref_q.get(k, 1e-4)) / (tot_q + 1e-4 * len(all_keys)) for k in all_keys}

    # Cross-Entropy: H(P, Q) = - sum P(x) * log2(Q(x))
    cross_entropy = -sum(p * math.log2(smooth_q[k]) for k, p in smooth_p.items())
    # KL-Divergence: D_KL(P || Q) = sum P(x) * log2(P(x) / Q(x))
    kl_div = sum(p * math.log2(p / smooth_q[k]) for k, p in smooth_p.items())

    return {
        'cross_entropy': round(cross_entropy, 4),
        'kl_divergence': round(max(0.0, kl_div), 4),
        'sample_chars': n
    }


def analyze_zipf_law(tokens: List[str]) -> Dict[str, Any]:
    """
    Fit Zipf's law log(freq) = -alpha * log(rank) + beta.
    Returns slope alpha, R^2 correlation, and top ranked tokens.
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip() and t.isalnum()]
    if not clean_tokens:
        return {'alpha': 0.0, 'r_squared': 0.0, 'ranks': [], 'freqs': []}

    counts = Counter(clean_tokens)
    sorted_counts = counts.most_common()

    ranks = np.arange(1, len(sorted_counts) + 1)
    freqs = np.array([cnt for _, cnt in sorted_counts])

    # Log-log transformation
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    corr_matrix = np.corrcoef(log_ranks, log_freqs)
    r_squared = (corr_matrix[0, 1]) ** 2 if corr_matrix.shape == (2, 2) else 0.0

    return {
        'alpha': round(abs(slope), 4),
        'r_squared': round(r_squared, 4),
        'top_tokens': sorted_counts[:20],
        'total_tokens': len(clean_tokens),
        'vocabulary_size': len(sorted_counts)
    }


def sukhotin_vowel_detector(text: str, top_vowels: int = 4) -> List[str]:
    """
    Implements Boris Sukhotin's algorithm (1962) to discover vowels in an unknown script
    based on character adjacency matrix.
    """
    chars = [c for c in text.lower() if c.isalnum()]
    if not chars:
        return []

    vocab = sorted(list(set(chars)))
    v_map = {c: i for i, c in enumerate(vocab)}
    m_len = len(vocab)
    
    matrix = np.zeros((m_len, m_len), dtype=np.float64)
    for i in range(len(chars) - 1):
        c1, c2 = chars[i], chars[i+1]
        matrix[v_map[c1], v_map[c2]] += 1
        matrix[v_map[c2], v_map[c1]] += 1

    vowels = []
    f = np.sum(matrix, axis=1) - 2 * np.diag(matrix)

    for _ in range(min(top_vowels, m_len)):
        best_char_idx = np.argmax(f)
        if f[best_char_idx] <= 0:
            break
        vowels.append(vocab[best_char_idx])
        for j in range(m_len):
            f[j] -= 2 * matrix[j, best_char_idx]
        f[best_char_idx] = -999999.0

    return vowels


def classify_currier_dialect(tokens: List[str]) -> Dict[str, Any]:
    """
    Distinguishes Currier Language A (Herbal/Pharma) from Currier Language B (Biological/Astrological).
    """
    clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
    if not clean_tokens:
        return {'dialect': 'Unknown', 'score_a': 0.0, 'score_b': 0.0}

    markers_a = {'daiin', 'dain', 'or', 'ar', 'al', 'ol', 'chol', 'chor', 'dal', 'dam'}
    markers_b = {'chedy', 'shedy', 'qokedy', 'qokeedy', 'qokain', 'chey', 'shey', 'otedy'}

    count_a = sum(1 for t in clean_tokens if t in markers_a or 'daiin' in t)
    count_b = sum(1 for t in clean_tokens if t in markers_b or 'eedy' in t)

    repeats = sum(1 for i in range(len(clean_tokens)-1) if clean_tokens[i] == clean_tokens[i+1])

    score_a = count_a / len(clean_tokens)
    score_b = (count_b + 2.0 * repeats) / len(clean_tokens)

    if score_a >= score_b:
        verdict = 'Currier Language A'
        confidence = min(0.99, round(0.5 + (score_a - score_b) * 2, 2))
    else:
        verdict = 'Currier Language B'
        confidence = min(0.99, round(0.5 + (score_b - score_a) * 2, 2))

    return {
        'dialect': verdict,
        'confidence': confidence,
        'score_a': round(score_a, 4),
        'score_b': round(score_b, 4),
        'repeat_count': repeats
    }
