"""
Voynich Slot Grammar & Morphological Decomposition
Implements the finite state morphological model of Voynichese:
Word = [Prefix] + [Medial / Gallows / Bench] + [Core / Vowel] + [Suffix / Coda]
"""

import re
from typing import Dict, Any, List, Optional

__all__ = [
    'PREFIXES', 'GALLOWS_BENCHES', 'CORES', 'SUFFIXES', 'GRAMMAR_PROFILES',
    'parse_token', 'parse_token_with_grammar', 'compare_grammars_on_tokens',
    'analyze_corpus_morphology'
]


# Canonical morphological inventories
PREFIXES = [
    'qok', 'qot', 'qop', 'qof', 'qo', 
    'so', 'sa', 'to', 'ta', 'po', 'ok', 'ot', 'op', 
    'ol', 'or', 'o', 'y', 'd', 's'
]

GALLOWS_BENCHES = [
    'cth', 'ckh', 'cph', 'cfh', 'tth', 'kkh',
    'ch', 'sh', 'k', 't', 'p', 'f'
]

CORES = [
    'eee', 'ee', 'e', 'o', 'a', 'ai', 'ey', 'eo', 'ea', 'i', 'u'
]

SUFFIXES = [
    'iiin', 'iin', 'in', 'dydy', 'dald', 'dy', 'dam', 'dan', 'dain',
    'ar', 'al', 'am', 'an', 'or', 'ol', 'y', 'r', 'l', 's', 'm', 'g', 'd'
]

# Standard Historical & Linguistic Grammar Profiles for Cross-Comparison
GRAMMAR_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "Klasik Voynich Slot Grameri (Stolfi / Tiltman)": {
        'prefixes': PREFIXES,
        'medials': GALLOWS_BENCHES,
        'cores': CORES,
        'suffixes': SUFFIXES
    },
    "Currier Dil A Grameri (Herbal / Astrological)": {
        'prefixes': ['o', 'ch', 'sh', 'd', 's', 'y', 'sa', 'so', 'ok', 'ot'],
        'medials': ['k', 't', 'p', 'f', 'ch', 'sh'],
        'cores': ['e', 'o', 'a', 'ai', 'ey', 'i'],
        'suffixes': ['dy', 'y', 'al', 'ar', 'ol', 'or', 's', 'm', 'd']
    },
    "Currier Dil B Grameri (Biological / Recipes)": {
        'prefixes': ['qo', 'qok', 'qot', 'qop', 'qof', 'cho', 'che', 'she', 'sho', 'ok', 'ot'],
        'medials': ['k', 't', 'p', 'f', 'ckh', 'cth', 'cph', 'cfh', 'ee'],
        'cores': ['ee', 'e', 'ai', 'ii', 'o', 'a', 'y'],
        'suffixes': ['daiin', 'aiin', 'iin', 'in', 'chedy', 'shedy', 'dy', 'ey', 'ar', 'al', 'am', 'ol', 'or']
    },
    "Latin Fleksiyonel Morfoloji Modeli (Kök + Gövde + Çekim Eki)": {
        'prefixes': ['con', 'sub', 'prae', 're', 'in', 'de', 'ex', 'dis', 'ad', 'ab'],
        'medials': ['radic', 'foli', 'herb', 'flor', 'fruct', 'semin', 'caul', 'calid', 'frigid', 'aqu', 'ole'],
        'cores': ['a', 'e', 'i', 'o', 'u', 'ae', 'oe'],
        'suffixes': ['ibus', 'orum', 'arum', 'ium', 'entia', 'alis', 'atus', 'osa', 'um', 'us', 'am', 'em', 'as', 'es', 'is', 'a', 'e', 'o', 'i']
    },
    "Sami Dilleri Üç Sessizli (Triliteral) Kök Modeli": {
        'prefixes': ['ha', 'be', 've', 'le', 'me', 'she', 'te', 'ye', 'ne'],
        'medials': ['ktb', 'sfr', 'shmr', 'mlk', 'qds', 'brk', 'rfe', 'zre', 'ntf'],
        'cores': ['a', 'i', 'u', 'o', 'e', 'ai', 'ei', 'ou'],
        'suffixes': ['im', 'ot', 'ah', 'nu', 'ka', 'chem', 'hem', 'i', 'a', 'at']
    }
}


def parse_token(word: str) -> Dict[str, Any]:
    """
    Deconstructs a Voynich token (in EVA) into its constituent morphological slots.
    Returns:
    - prefix
    - medial (gallows or bench)
    - core (vowel/base)
    - suffix
    - is_valid: whether the word could be cleanly parsed into standard slots
    - remainder: any unparsed characters
    """
    cleaned = word.lower().strip()
    cleaned = re.sub(r'[{}\$!=?#@0-9;%*+,<>\-\.]', '', cleaned)
    
    if not cleaned:
        return {'raw': word, 'prefix': None, 'medial': None, 'core': None, 'suffix': None, 'is_valid': False, 'remainder': ''}

    w = cleaned
    prefix = None
    medial = None
    core = None
    suffix = None

    # Handle 'qo-' prefix specifically so gallows 'k', 't', 'p' are assigned to medial
    if w.startswith('qo') and len(w) > 2 and w[2] in ['k', 't', 'p', 'f']:
        prefix = 'qo'
        w = w[2:]
    else:
        for p in sorted(PREFIXES, key=len, reverse=True):
            if w.startswith(p):
                # Ensure we don't consume the entire word if it's longer
                if len(w) > len(p) or len(w) <= 2:
                    prefix = p
                    w = w[len(p):]
                    break

    # Match Suffix (from the end, longest match first)
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if w.endswith(s):
            suffix = s
            w = w[:-len(s)]
            break

    # Match Medial / Gallows / Bench
    for m in sorted(GALLOWS_BENCHES, key=len, reverse=True):
        if m in w:
            idx = w.find(m)
            medial = m
            w = w[:idx] + w[idx+len(m):]
            break

    # What remains is candidate core
    for c in sorted(CORES, key=len, reverse=True):
        if c == w or c in w:
            core = c
            w = w.replace(c, '', 1)
            break

    remainder = w
    is_valid = (len(remainder) == 0) and any([prefix, medial, core, suffix])

    return {
        'raw': word,
        'cleaned': cleaned,
        'prefix': prefix,
        'medial': medial,
        'core': core,
        'suffix': suffix,
        'remainder': remainder,
        'is_valid': is_valid
    }


def analyze_corpus_morphology(tokens: List[str]) -> Dict[str, Any]:
    """
    Analyze a collection of tokens and compute slot occupancy frequencies
    and grammar adherence rate.
    """
    parsed_list = [parse_token(t) for t in tokens if t.strip()]
    if not parsed_list:
        return {
            'total_tokens': 0, 'valid_tokens': 0, 'valid_rate': 0.0,
            'prefix_frequencies': {}, 'medial_frequencies': {},
            'core_frequencies': {}, 'suffix_frequencies': {},
            'parsed_samples': []
        }

    valid_count = sum(1 for p in parsed_list if p['is_valid'])
    valid_rate = valid_count / len(parsed_list)

    prefix_counts: Dict[str, int] = {}
    medial_counts: Dict[str, int] = {}
    core_counts: Dict[str, int] = {}
    suffix_counts: Dict[str, int] = {}

    for p in parsed_list:
        if p['prefix']:
            prefix_counts[p['prefix']] = prefix_counts.get(p['prefix'], 0) + 1
        if p['medial']:
            medial_counts[p['medial']] = medial_counts.get(p['medial'], 0) + 1
        if p['core']:
            core_counts[p['core']] = core_counts.get(p['core'], 0) + 1
        if p['suffix']:
            suffix_counts[p['suffix']] = suffix_counts.get(p['suffix'], 0) + 1

    return {
        'total_tokens': len(parsed_list),
        'valid_tokens': valid_count,
        'valid_rate': round(valid_rate, 4),
        'prefix_frequencies': dict(sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)),
        'medial_frequencies': dict(sorted(medial_counts.items(), key=lambda x: x[1], reverse=True)),
        'core_frequencies': dict(sorted(core_counts.items(), key=lambda x: x[1], reverse=True)),
        'suffix_frequencies': dict(sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True)),
        'parsed_samples': parsed_list[:50]
    }


def parse_token_with_grammar(word: str, grammar: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Deconstructs a word using an arbitrary specified grammar dictionary
    with keys 'prefixes', 'medials', 'cores', 'suffixes'.
    """
    cleaned = word.lower().strip()
    cleaned = re.sub(r'[{}\$!=?#@0-9;%*+,<>\-\.]', '', cleaned)
    if not cleaned:
        return {'raw': word, 'prefix': None, 'medial': None, 'core': None, 'suffix': None, 'is_valid': False, 'remainder': ''}

    w = cleaned
    prefix = None
    medial = None
    core = None
    suffix = None

    prefixes = grammar.get('prefixes', [])
    medials = grammar.get('medials', [])
    cores = grammar.get('cores', [])
    suffixes = grammar.get('suffixes', [])

    # Match Prefix (longest match first)
    for p in sorted(prefixes, key=len, reverse=True):
        if w.startswith(p):
            if len(w) > len(p) or len(w) <= 2:
                prefix = p
                w = w[len(p):]
                break

    # Match Suffix (longest match first from right)
    for s in sorted(suffixes, key=len, reverse=True):
        if w.endswith(s):
            suffix = s
            w = w[:-len(s)]
            break

    # Match Medial
    for m in sorted(medials, key=len, reverse=True):
        if m in w:
            idx = w.find(m)
            medial = m
            w = w[:idx] + w[idx+len(m):]
            break

    # Match Core
    for c in sorted(cores, key=len, reverse=True):
        if c == w or c in w:
            core = c
            w = w.replace(c, '', 1)
            break

    remainder = w
    is_valid = (len(remainder) == 0) and any([prefix, medial, core, suffix])

    return {
        'raw': word,
        'cleaned': cleaned,
        'prefix': prefix,
        'medial': medial,
        'core': core,
        'suffix': suffix,
        'remainder': remainder,
        'is_valid': is_valid
    }


def compare_grammars_on_tokens(tokens: List[str],
                               selected_profiles: Optional[List[str]] = None,
                               custom_grammar: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Evaluates and ranks multiple grammar models against a token collection.
    Returns adherence rate, slot distributions, and comparison rankings.
    """
    profiles_to_test = dict(GRAMMAR_PROFILES)
    if custom_grammar:
        profiles_to_test["Özel / Kullanıcı Tanımlı Gramer"] = custom_grammar

    if selected_profiles:
        profiles_to_test = {k: v for k, v in profiles_to_test.items() if k in selected_profiles}

    clean_tokens = [t for t in tokens if t.strip()]
    if not clean_tokens:
        return {'results': [], 'rankings': [], 'best_grammar': None}

    comparison_results = []
    for gname, grammar in profiles_to_test.items():
        parsed = [parse_token_with_grammar(t, grammar) for t in clean_tokens]
        valid_count = sum(1 for p in parsed if p['is_valid'])
        rate = valid_count / len(clean_tokens)

        # Slot occupancy counts
        p_cnt = sum(1 for p in parsed if p['prefix'])
        m_cnt = sum(1 for p in parsed if p['medial'])
        c_cnt = sum(1 for p in parsed if p['core'])
        s_cnt = sum(1 for p in parsed if p['suffix'])

        comparison_results.append({
            'grammar': gname,
            'adherence_rate': round(rate * 100.0, 1),
            'valid_tokens': valid_count,
            'total_tokens': len(clean_tokens),
            'prefix_coverage': round((p_cnt / len(clean_tokens)) * 100.0, 1),
            'medial_coverage': round((m_cnt / len(clean_tokens)) * 100.0, 1),
            'core_coverage': round((c_cnt / len(clean_tokens)) * 100.0, 1),
            'suffix_coverage': round((s_cnt / len(clean_tokens)) * 100.0, 1),
            'samples': parsed[:10]
        })

    comparison_results.sort(key=lambda r: -r['adherence_rate'])
    best_grammar = comparison_results[0]['grammar'] if comparison_results else None

    return {
        'results': comparison_results,
        'best_grammar': best_grammar,
        'tested_count': len(comparison_results)
    }
