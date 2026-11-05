"""
Historical Cryptographic Hypothesis Simulator
---------------------------------------------
Implements historical Renaissance and Medieval cipher hypotheses for Voynich analysis:
1. Cardan Grille (Cardano Izgarası) mask simulation with 4 rotations (0°, 90°, 180°, 270°).
2. Verbose / Homophonic substitution analysis and equivalence clustering.
3. Alberti cipher disk & Vigenère periodic solver:
   - Kasiski examination (n-gram distance GCD analysis)
   - Periodic autocorrelation & Friedman test for key length detection.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.crypto_studio import MEDIEVAL_BOTANICAL_LEXICON


# -----------------------------------------------------------------------------
# 1. CARDAN GRILLE SIMULATOR (CARDANO IZGARA MASKESİ)
# -----------------------------------------------------------------------------
class CardanGrilleSimulator:
    """
    Simulates Cardan Grille (Cardano Izgarası) over a text matrix.
    A valid K x K Cardan grille has K^2 / 4 apertures such that rotating
    through 4 quadrants (0°, 90°, 180°, 270°) covers every cell exactly once.
    """

    def __init__(self, k: int = 6):
        if k % 2 != 0:
            k = 6
        self.k = k
        self.total_cells = k * k
        self.apertures_count = self.total_cells // 4

    def generate_canonical_grille(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a valid Cardan grille mask of shape (K, K) with boolean True for apertures.
        Each of the K^2/4 orbits under 90° rotation has exactly one hole.
        """
        rng = np.random.RandomState(seed)
        mask = np.zeros((self.k, self.k), dtype=bool)
        half = self.k // 2

        # In a 2D grid, coordinate (r, c) under 90° rotation maps to:
        # (r, c) -> (c, k - 1 - r) -> (k - 1 - r, k - 1 - c) -> (k - 1 - c, r)
        for r in range(half):
            for c in range(half):
                orbit = [
                    (r, c),
                    (c, self.k - 1 - r),
                    (self.k - 1 - r, self.k - 1 - c),
                    (self.k - 1 - c, r)
                ]
                choice = rng.randint(0, 4)
                chosen_r, chosen_c = orbit[choice]
                mask[chosen_r, chosen_c] = True

        return mask

    def apply_grille_to_matrix(self, char_matrix: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Applies mask with 4 rotations (0°, 90°, 180°, 270°) to an (N, M) matrix.
        Extracts letters revealed at each turn.
        """
        H, W = char_matrix.shape
        kh, kw = mask.shape
        extracted_rounds = {0: [], 90: [], 180: [], 270: []}

        # Slide grille across matrix in steps of K
        for top in range(0, H - kh + 1, kh):
            for left in range(0, W - kw + 1, kw):
                sub = char_matrix[top:top+kh, left:left+kw]
                cur_mask = mask.copy()
                for angle in [0, 90, 180, 270]:
                    if angle > 0:
                        cur_mask = np.rot90(cur_mask, -1)  # rotate 90 deg clockwise
                    chars = sub[cur_mask]
                    extracted_rounds[angle].extend(chars.tolist())

        full_extracted = (
            extracted_rounds[0] +
            extracted_rounds[90] +
            extracted_rounds[180] +
            extracted_rounds[270]
        )
        text_extracted = "".join(full_extracted).lower()

        # Score extracted text against dictionary roots
        words_found = []
        for word in MEDIEVAL_BOTANICAL_LEXICON:
            if len(word) >= 4 and word in text_extracted:
                words_found.append(word)

        return {
            'text_extracted': text_extracted,
            'rounds': {
                '0_deg': "".join(extracted_rounds[0]),
                '90_deg': "".join(extracted_rounds[90]),
                '180_deg': "".join(extracted_rounds[180]),
                '270_deg': "".join(extracted_rounds[270])
            },
            'length': len(text_extracted),
            'matched_roots': list(set(words_found)),
            'hit_count': len(words_found)
        }

    def simulate_search(self, text: str, n_trials: int = 40) -> Dict[str, Any]:
        """
        Stochastically tests multiple Cardan grilles across the page text.
        Finds the aperture arrangement yielding the highest linguistic score.
        """
        chars = [c for c in text.lower() if c.isalnum()]
        if len(chars) < self.total_cells:
            # Pad if needed
            chars = chars + ['o'] * (self.total_cells - len(chars))

        # Shape into rectangle
        n_chars = len(chars)
        n_blocks = max(1, n_chars // self.total_cells)
        fit_len = n_blocks * self.total_cells
        trimmed = chars[:fit_len]
        
        # Grid shape (k * n_blocks, k)
        char_matrix = np.array(trimmed).reshape((self.k * n_blocks, self.k))

        best_result = None
        best_score = -1.0

        for trial in range(n_trials):
            mask = self.generate_canonical_grille(seed=trial * 37 + 11)
            res = self.apply_grille_to_matrix(char_matrix, mask)
            # Score: hit count + vowel ratio plausibility
            vowels = sum(1 for c in res['text_extracted'] if c in 'aeiou')
            v_ratio = vowels / max(len(res['text_extracted']), 1)
            score = res['hit_count'] * 10.0 + max(0, 5.0 - abs(v_ratio - 0.44) * 15.0)

            if score > best_score:
                best_score = score
                best_result = {
                    'trial_id': trial,
                    'score': round(score, 2),
                    'matched_roots': res['matched_roots'],
                    'vowel_ratio': round(v_ratio * 100, 1),
                    'extracted_sample': res['text_extracted'][:80],
                    'rounds': res['rounds'],
                    'grille_mask': mask.astype(int).tolist()
                }

        return best_result or {
            'trial_id': 0, 'score': 0.0, 'matched_roots': [],
            'vowel_ratio': 0.0, 'extracted_sample': '', 'rounds': {}, 'grille_mask': []
        }


# -----------------------------------------------------------------------------
# 2. VERBOSE / HOMOPHONIC SUBSTITUTION ENGINE
# -----------------------------------------------------------------------------
class HomophonicCipherEngine:
    """
    Analyzes verbose / homophonic substitution:
    - Multiple Voynich glyphs / compounds mapping to a single plaintext character.
    - Tests if clustering high-frequency Voynich tokens reduces conditional entropy.
    """

    def analyze_homophonic_clusters(self, tokens: List[str], top_n: int = 15) -> Dict[str, Any]:
        clean_tokens = [t.lower().strip() for t in tokens if t.strip()]
        if len(clean_tokens) < 10:
            return {'clusters': [], 'entropy_reduction': 0.0}

        counts = Counter(clean_tokens)
        most_common = [w for w, _ in counts.most_common(top_n)]

        # Context matrix: for each target word, vector of preceding word counts
        prec_counts: Dict[str, Counter] = {w: Counter() for w in most_common}
        succ_counts: Dict[str, Counter] = {w: Counter() for w in most_common}

        for i in range(len(clean_tokens) - 1):
            w1, w2 = clean_tokens[i], clean_tokens[i+1]
            if w2 in prec_counts:
                prec_counts[w2][w1] += 1
            if w1 in succ_counts:
                succ_counts[w1][w2] += 1

        # Cosine similarity between contextual vectors of word pairs
        candidates = []
        for i in range(len(most_common)):
            for j in range(i + 1, len(most_common)):
                w_a, w_b = most_common[i], most_common[j]
                # Combined context overlap
                p_a, p_b = prec_counts[w_a], prec_counts[w_b]
                all_p = set(p_a.keys()).union(set(p_b.keys()))
                if not all_p:
                    continue
                va = np.array([p_a.get(k, 0) for k in all_p], dtype=float)
                vb = np.array([p_b.get(k, 0) for k in all_p], dtype=float)
                norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(va, vb) / (norm_a * norm_b))
                    if sim > 0.40:
                        candidates.append({
                            'token_1': w_a,
                            'token_2': w_b,
                            'similarity': round(sim, 3),
                            'freq_1': counts[w_a],
                            'freq_2': counts[w_b],
                            'hypothesis': f"'{w_a}' ve '{w_b}' ayni ses/harf icin homofonik varyant olabilir."
                        })

        candidates.sort(key=lambda x: -x['similarity'])

        # Calculate hypothetical entropy reduction if top candidate pair is merged
        orig_text = "".join(clean_tokens)
        c_orig = Counter(orig_text)
        h_orig = -sum((cnt / len(orig_text)) * math.log2(cnt / len(orig_text)) for cnt in c_orig.values())

        if candidates:
            top_pair = (candidates[0]['token_1'], candidates[0]['token_2'])
            merged_tokens = [top_pair[0] if t == top_pair[1] else t for t in clean_tokens]
            merged_text = "".join(merged_tokens)
            c_merged = Counter(merged_text)
            h_merged = -sum((cnt / len(merged_text)) * math.log2(cnt / len(merged_text)) for cnt in c_merged.values())
            delta_h = round(h_orig - h_merged, 4)
        else:
            delta_h = 0.0

        return {
            'homophonic_candidates': candidates[:10],
            'baseline_char_entropy': round(h_orig, 4),
            'entropy_reduction': delta_h
        }


# -----------------------------------------------------------------------------
# 3. ALBERTI DISK & VIGENÈRE POLYALPHABETIC SOLVER
# -----------------------------------------------------------------------------
class PolyalphabeticSolver:
    """
    Kasiski examination and Periodic Autocorrelation for testing
    periodic polyalphabetic (Alberti / Vigenère / Trithemius) ciphers.
    """

    def kasiski_examination(self, text: str, min_len: int = 3, max_len: int = 5) -> Dict[str, Any]:
        """
        Finds repeated n-grams and computes greatest common divisors (GCD) of distances.
        """
        clean_text = "".join(c for c in text.lower() if c.isalnum())
        n = len(clean_text)
        if n < 30:
            return {'repeated_ngrams': [], 'probable_key_lengths': []}

        # Track positions of all n-grams
        ngram_positions: Dict[str, List[int]] = defaultdict(list)
        for length in range(min_len, max_len + 1):
            for i in range(n - length + 1):
                sub = clean_text[i:i+length]
                ngram_positions[sub].append(i)

        # Retain only n-grams occurring at least twice
        repeated = {k: v for k, v in ngram_positions.items() if len(v) >= 2}

        # Compute pairwise distance differences
        distances = []
        repeat_records = []
        for ngram, pos_list in sorted(repeated.items(), key=lambda x: -len(x[1])):
            if len(ngram) < min_len:
                continue
            diffs = [pos_list[j] - pos_list[j-1] for j in range(1, len(pos_list))]
            distances.extend(diffs)
            repeat_records.append({
                'ngram': ngram,
                'count': len(pos_list),
                'positions': pos_list[:5],
                'distances': diffs[:5]
            })

        # Factorize distances (candidate periods from 2 to 12)
        factor_counts = Counter()
        for d in distances:
            for factor in range(2, 13):
                if d % factor == 0:
                    factor_counts[factor] += 1

        top_factors = [
            {'key_length': k, 'gcd_hits': count, 'score': round(count / max(len(distances), 1), 3)}
            for k, count in factor_counts.most_common(6)
        ]

        return {
            'total_repeats_found': len(repeat_records),
            'top_repeated_ngrams': repeat_records[:12],
            'probable_key_lengths': top_factors
        }

    def autocorrelation_test(self, text: str, max_shift: int = 15) -> Dict[str, Any]:
        """
        Friedman autocorrelation test: calculates coincidence rate for shifts d = 1..max_shift.
        Periodic peaks at d, 2d, 3d indicate periodic polyalphabetic key length d.
        """
        clean_text = "".join(c for c in text.lower() if c.isalnum())
        n = len(clean_text)
        if n < max_shift + 5:
            return {'shifts': [], 'peaks': []}

        coincidences = []
        for shift in range(1, max_shift + 1):
            matches = sum(1 for i in range(n - shift) if clean_text[i] == clean_text[i + shift])
            rate = matches / (n - shift)
            coincidences.append({'shift': shift, 'coincidence_rate': round(rate, 4)})

        rates = [c['coincidence_rate'] for c in coincidences]
        mean_rate = float(np.mean(rates))
        std_rate = float(np.std(rates)) if len(rates) > 1 else 0.01

        # Detect peaks (> 1.25 std above mean)
        peaks = [
            c['shift'] for c in coincidences
            if c['coincidence_rate'] > mean_rate + 1.25 * std_rate
        ]

        return {
            'mean_coincidence': round(mean_rate, 4),
            'shifts': coincidences,
            'periodic_peaks': peaks,
            'has_periodic_structure': len(peaks) > 0
        }
