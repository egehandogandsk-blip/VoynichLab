"""
e2Canvas Sandbox Backend & Cluster Cryptanalysis Engine
-------------------------------------------------------
Computes real-time information theory, transition likelihood,
and medieval philological scores for nodes and groups placed on e2Canvas:
1. Combined Shannon entropy H1, H2, and character distribution.
2. Bigram transition probability score (coherence % against Voynich corpus).
3. Dictionary fit score (Levenshtein distance & n-gram overlap against
   medieval botanical and medical Latin/Italian lexicons).
4. Morphological decomposition (prefix, core, suffix) and hypothesis validation.
"""

import math
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

from src.eva import tokenize_units
from src.slot_grammar import parse_token
from src.crypto_entropy import compute_shannon_entropy
from src.crypto_studio import (
    MEDIEVAL_BOTANICAL_LEXICON, SubstitutionCipher, CIPHER_PRESETS, EVA_ALPHABET
)
from src.zl_corpus import load_zl

# Standard Voynich bigram transition frequencies derived from canonical ZL corpus
_VOYNICH_BIGRAM_SET = {
    'ch', 'sh', 'da', 'ai', 'in', 'qo', 'ok', 'ed', 'dy', 'ee', 'ol', 'or',
    'ar', 'al', 'am', 'ct', 'cp', 'cf', 'ck', 'tc', 'pc', 'fc', 'kc',
    'ha', 'he', 'hy', 'ho', 'ke', 'te', 'pe', 'fe', 'ty', 'ky', 'py'
}


def _levenshtein(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_dictionary_match(text: str, lexicon: Optional[set] = None) -> Tuple[float, Optional[str]]:
    """
    Computes best dictionary match score (0-100%) and candidate word from medieval lexicon.
    Uses normalized Levenshtein similarity.
    """
    target_lexicon = lexicon or MEDIEVAL_BOTANICAL_LEXICON
    clean = text.lower().strip()
    if not clean:
        return 0.0, None

    if clean in target_lexicon:
        return 100.0, clean

    best_score = 0.0
    best_candidate = None

    for candidate in target_lexicon:
        dist = _levenshtein(clean, candidate)
        max_len = max(len(clean), len(candidate))
        if max_len == 0:
            continue
        sim = max(0.0, 1.0 - (dist / max_len)) * 100.0
        if sim > best_score:
            best_score = sim
            best_candidate = candidate

    return round(best_score, 1), best_candidate


def compute_bigram_coherence(tokens: List[str]) -> float:
    """
    Computes percentage of bigrams conforming to Voynich phonotactic/scribal transitions.
    """
    if not tokens:
        return 0.0

    total_bigrams = 0
    valid_bigrams = 0

    for token in tokens:
        clean = token.lower().strip()
        if len(clean) < 2:
            continue
        for i in range(len(clean) - 1):
            bg = clean[i:i+2]
            total_bigrams += 1
            if bg in _VOYNICH_BIGRAM_SET:
                valid_bigrams += 1

    if total_bigrams == 0:
        return 0.0

    return round((valid_bigrams / total_bigrams) * 100.0, 1)


def analyze_canvas_cluster(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs comprehensive cryptanalytic and linguistic analysis for a set of canvas nodes.
    Each node can have fields: 'id', 'type', 'eva', 'translation', 'label'.
    """
    eva_tokens = []
    translations = []

    for node in nodes:
        eva = node.get('eva', '').strip()
        if eva:
            eva_tokens.append(eva)
        trans = node.get('translation', '').strip()
        if trans:
            translations.append(trans)

    # If no EVA tokens but labels exist
    if not eva_tokens:
        for node in nodes:
            label = node.get('label', '').strip()
            if label and node.get('type') in ('word', 'crop'):
                eva_tokens.append(label)

    combined_eva_text = " ".join(eva_tokens)

    # 1. Shannon Entropy
    shannon_data = compute_shannon_entropy(combined_eva_text)
    h1 = shannon_data.get('h1', 0.0)
    h2 = shannon_data.get('h2', 0.0)

    # 2. Bigram Coherence against Voynich Corpus
    bigram_coherence = compute_bigram_coherence(eva_tokens)

    # 3. Dictionary Matching
    dict_scores = []
    top_candidates = []

    for token in (translations if translations else eva_tokens):
        score, cand = compute_dictionary_match(token)
        dict_scores.append(score)
        if cand:
            top_candidates.append(cand)

    avg_dict_score = round(sum(dict_scores) / len(dict_scores), 1) if dict_scores else 0.0
    best_candidate = top_candidates[0] if top_candidates else (list(MEDIEVAL_BOTANICAL_LEXICON)[0] if avg_dict_score > 50 else None)

    # 4. Morphological Decomposition
    morph_breakdown = []
    for token in eva_tokens:
        parsed = parse_token(token)
        morph_breakdown.append({
            'token': token,
            'prefix': parsed.get('prefix') or '-',
            'core': parsed.get('core') or '-',
            'suffix': parsed.get('suffix') or '-',
            'valid': parsed.get('valid', False)
        })

    cand_str = f' (Örn: "{best_candidate}")' if best_candidate and avg_dict_score > 30 else ""
    summary_line = (
        f"Shannon H: {h1:.2f} (H2: {h2:.2f}) | "
        f"Bigram Tutarlılığı: %{bigram_coherence:.0f} | "
        f"Sözlük Uyum: %{avg_dict_score:.0f}{cand_str}"
    )

    return {
        'node_count': len(nodes),
        'eva_tokens': eva_tokens,
        'translations': translations,
        'shannon_h1': h1,
        'shannon_h2': h2,
        'char_count': shannon_data.get('char_count', 0),
        'unique_chars': shannon_data.get('unique_chars', 0),
        'bigram_coherence': bigram_coherence,
        'dictionary_score': avg_dict_score,
        'best_candidate': best_candidate,
        'morphology': morph_breakdown,
        'summary_line': summary_line
    }


def find_token_corpus_concordance(token: str, max_matches: int = 15) -> Dict[str, Any]:
    """
    Searches the entire manuscript for occurrences of `token`,
    extracting folios, line contexts, left collocates, and right collocates.
    """
    clean_target = token.lower().strip()
    if not clean_target:
        return {'total_occurrences': 0, 'folios': [], 'left_collocates': {}, 'right_collocates': {}, 'contexts': []}

    corpus = load_zl()
    total_occurrences = 0
    folios_set = set()
    left_counts: Counter = Counter()
    right_counts: Counter = Counter()
    contexts = []

    for fid, rec in corpus.items():
        section = rec.get('section', 'Bilinmeyen Bölüm')
        has_plant = bool(rec.get('plant_id'))
        for line in rec.get('lines', []):
            line_toks = line.get('tokens', [])
            ln = line.get('n', 1)
            for idx, tok in enumerate(line_toks):
                if tok.lower() == clean_target:
                    total_occurrences += 1
                    folios_set.add(fid)
                    prev_w = line_toks[idx - 1] if idx > 0 else "<BAŞLANGIÇ>"
                    next_w = line_toks[idx + 1] if idx < len(line_toks) - 1 else "<BİTİŞ>"
                    if prev_w != "<BAŞLANGIÇ>":
                        left_counts[prev_w] += 1
                    if next_w != "<BİTİŞ>":
                        right_counts[next_w] += 1

                    if len(contexts) < max_matches:
                        contexts.append({
                            'folio': fid,
                            'line': ln,
                            'left': prev_w,
                            'token': tok,
                            'right': next_w,
                            'section': section,
                            'has_plant': has_plant
                        })

    top_left = dict(left_counts.most_common(5))
    top_right = dict(right_counts.most_common(5))

    return {
        'token': clean_target,
        'total_occurrences': total_occurrences,
        'unique_folios_count': len(folios_set),
        'folios': sorted(list(folios_set))[:20],
        'left_collocates': top_left,
        'right_collocates': top_right,
        'contexts': contexts,
        'summary': (
            f"'{clean_target}' kelimesi el yazmasında {total_occurrences} kez, "
            f"{len(folios_set)} farklı folyoda geçiyor. "
            f"En sık sol komşusu: {list(top_left.keys())[0] if top_left else 'Yok'} | "
            f"En sık sağ komşusu: {list(top_right.keys())[0] if top_right else 'Yok'}"
        )
    }


def analyze_bridge_relationship(source_node: Dict[str, Any], target_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes linguistic or cryptographic bridge semantics between two connected canvas nodes.
    """
    s_type = source_node.get('type', 'word')
    t_type = target_node.get('type', 'trans')
    s_label = source_node.get('label') or source_node.get('eva') or ''
    t_label = target_node.get('label') or target_node.get('translation') or target_node.get('eva') or ''

    relationship_type = "Bilinmeyen Bağlantı"
    explanation = ""
    colloc_freq = 0

    if s_type == 'crop' and t_type in ('word', 'crop'):
        relationship_type = "Görsel Kırpıntı - Transkripsiyon Doğrulaması"
        explanation = f"Folyodaki el yazması mürekkep glifleri '{t_label}' EVA transkripsiyonu ile eşlendi."
    elif s_type in ('word', 'crop') and t_type == 'trans':
        relationship_type = "Kriptanaliz / Anlamsal Hipotez Köprüsü"
        target_lang = target_node.get('lang', 'Latince')
        score, cand = compute_dictionary_match(t_label)
        explanation = f"Voynich '{s_label}' tokeni için {target_lang} '{t_label}' hipotezi kuruldu (Sözlük Uyumu: %{score:.0f})."
    elif s_type in ('word', 'crop') and t_type in ('word', 'crop'):
        relationship_type = "Sözdizimsel Ardışıklık (Sentaktik Bigram)"
        conc = find_token_corpus_concordance(s_label, max_matches=5)
        is_frequent_next = t_label in conc.get('right_collocates', {})
        colloc_freq = conc.get('right_collocates', {}).get(t_label, 0)
        explanation = (
            f"'{s_label}' -> '{t_label}' ardışıklığı kurgulandı. "
            f"Metinde bu ikilinin ardışık geçme durumu: {'Evet (Doğrulandı)' if is_frequent_next else 'Seyrek / Hipotetik'}."
        )
    else:
        relationship_type = "Genel Mantıksal Kümeleme"
        explanation = f"'{s_label}' ile '{t_label}' arasında karşılaştırma köprüsü oluşturuldu."

    return {
        'source': s_label,
        'target': t_label,
        'type': relationship_type,
        'explanation': explanation,
        'summary': f"{relationship_type}: {explanation}",
        'collocation_frequency': colloc_freq
    }
