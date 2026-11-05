"""
Voynich Crib & Botanical Lexicon Matching Engine
Analyzes letter patterns (isomorphs), correlates illustration features with medieval
herbal vocabularies (Dioscorides, Herbarius, Macer Floridus), and tests substitution hypotheses.
"""

from typing import List, Dict, Any, Tuple


# Medieval botanical candidates curated for Voynich herbal folios
FOLIO_BOTANICAL_CRIBS = {
    '11r': [
        {'name': 'Silene', 'latin': 'silene acaulis', 'family': 'Caryophyllaceae', 'german': 'leimkraut'},
        {'name': 'Saponaria', 'latin': 'saponaria officinalis', 'family': 'Caryophyllaceae', 'german': 'seifenkraut'},
        {'name': 'Centaurium', 'latin': 'centaurium erythraea', 'family': 'Gentianaceae', 'german': 'tausendgueldenkraut'}
    ],
    '11v': [
        {'name': 'Leonurus', 'latin': 'leonurus cardiaca', 'family': 'Lamiaceae', 'german': 'herzgespann'},
        {'name': 'Leonotis', 'latin': 'leonotis leonurus', 'family': 'Lamiaceae', 'german': 'loewenohr'},
        {'name': 'Carduus', 'latin': 'carduus nutans', 'family': 'Asteraceae', 'german': 'distel'}
    ],
    '16r': [
        {'name': 'Cannabis', 'latin': 'cannabis sativa', 'family': 'Cannabaceae', 'german': 'hanf'},
        {'name': 'Canapis', 'latin': 'canapis', 'family': 'Cannabaceae', 'italian': 'canapa'},
        {'name': 'Urtica', 'latin': 'urtica dioica', 'family': 'Urticaceae', 'german': 'brennnessel'}
    ],
    '18r': [
        {'name': 'Cyanus', 'latin': 'cyanus segetum', 'family': 'Asteraceae', 'german': 'kornblume'},
        {'name': 'Centaurea', 'latin': 'centaurea cyanus', 'family': 'Asteraceae', 'italian': 'centaurea'},
        {'name': 'Scabiosa', 'latin': 'scabiosa columbaria', 'family': 'Caprifoliaceae', 'german': 'skabiose'}
    ]
}

# Common medieval pharmacy and herbal recipe crib formula words
MEDIEVAL_FORMULA_CRIBS = [
    'radix', 'folia', 'herba', 'flos', 'semen', 'cortex', 'succus', 
    'aqua', 'oleum', 'pulvis', 'balsamum', 'contra', 'valet', 'sanat', 'purgat'
]


def compute_isomorphic_pattern(word: str) -> str:
    """
    Computes numerical isomorphic pattern of a word based on letter repetitions.
    Example: 'cannabis' -> '0.1.2.2.1.3.4'
             'daiin'    -> '0.1.2.2.3'
    """
    cleaned = ''.join(c for c in word.lower() if c.isalnum())
    mapping = {}
    pattern = []
    curr_id = 0
    for char in cleaned:
        if char not in mapping:
            mapping[char] = str(curr_id)
            curr_id += 1
        pattern.append(mapping[char])
    return '.'.join(pattern)


def search_isomorphic_matches(voynich_tokens: List[str], candidate_words: List[str]) -> List[Dict[str, Any]]:
    """
    Finds tokens in the Voynich text that share identical letter repetition patterns with candidate crib words.
    """
    # Build dictionary of candidate patterns
    cand_dict: Dict[str, List[str]] = {}
    for cand in candidate_words:
        pat = compute_isomorphic_pattern(cand)
        cand_dict.setdefault(pat, []).append(cand)

    matches = []
    for token in set(voynich_tokens):
        t_pat = compute_isomorphic_pattern(token)
        if t_pat in cand_dict:
            matches.append({
                'token': token,
                'pattern': t_pat,
                'matched_candidates': cand_dict[t_pat]
            })

    return matches


def match_folio_cribs(folio_id: str, voynich_tokens: List[str]) -> Dict[str, Any]:
    """
    Check a folio's tokens against botanical candidates and medieval pharmacy formula cribs.
    """
    clean_id = folio_id.lower().replace('f', '').strip()
    botanical = FOLIO_BOTANICAL_CRIBS.get(clean_id, [])
    all_candidates = [b['name'].lower() for b in botanical] + [b['latin'].split()[0].lower() for b in botanical] + MEDIEVAL_FORMULA_CRIBS

    iso_matches = search_isomorphic_matches(voynich_tokens, all_candidates)

    return {
        'folio_id': clean_id,
        'botanical_candidates': botanical,
        'tested_cribs_count': len(all_candidates),
        'isomorphic_matches': iso_matches
    }
