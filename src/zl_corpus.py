"""
ZL (Zandbergen-Landini) IVTFF transcription parser.
Parses the whole transcript once into a structured, cached object so every module
(pipeline, corpus builder, studio, archive) works from the same clean tokens.
"""

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRANSCRIPT = os.path.join(BASE_DIR, 'data', 'transcripts', 'ZL3b-n.txt')

ILLUSTRATION_TYPES = {
    'H': 'Bitkiler (Herbal)', 'A': 'Astronomi / Astroloji', 'Z': 'Zodyak',
    'B': 'Biyolojik (Banyo)', 'C': 'Kozmolojik', 'P': 'Eczacılık (Pharmaceutical)',
    'S': 'Yıldızlı Reçeteler (Stars)', 'T': 'Yalnızca Metin', 'X': 'Diğer / Belirsiz',
}

_FOLIO_TAG = re.compile(r'^<f(\w+)>\s*(?:<!(.*)>)?\s*$')
_LINE_TAG = re.compile(r'^<f(\w+)\.(\d+)(?:;\w+)?,([^>]*)>\s*(.*)$')


def clean_line(raw: str) -> List[str]:
    """Convert one raw IVTFF text line to a list of clean EVA tokens."""
    s = raw.replace('<->', '.')            # figure-interrupted line: left/right parts
    s = re.sub(r'<![^>]*>', '', s)         # inline comments
    s = re.sub(r'<[^>]*>', '', s)          # remaining tags (<%>, <$>, <->)
    s = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', s)   # [a:b] alternatives -> first reading
    s = re.sub(r'\{([^}]*)\}', r'\1', s)   # {ct} ligature -> inner letters
    s = re.sub(r'@\d+;', '?', s)           # unknown glyph codes
    s = s.replace('-', '.')
    s = re.sub(r"[^a-z.,?' ]", '', s.lower())
    s = s.replace("'", '')
    return [t for t in re.split(r'[.,\s]+', s) if t]


@lru_cache(maxsize=4)
def load_zl(path: str = DEFAULT_TRANSCRIPT) -> Dict[str, Dict[str, Any]]:
    """Return {folio_id: {meta, lines:[{n, locus, tokens, raw}], comments:[...]}} (insertion ordered)."""
    corpus: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return corpus
    current: Optional[Dict[str, Any]] = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line.strip():
                continue
            m = _FOLIO_TAG.match(line)
            if m:
                fid, tags = m.group(1), m.group(2) or ''
                meta = dict(re.findall(r'\$(\w)=(\w)', tags))
                current = {
                    'folio': fid,
                    'meta': meta,
                    'language': meta.get('L', ''),
                    'hand': meta.get('H', ''),
                    'illustration': meta.get('I', ''),
                    'section': ILLUSTRATION_TYPES.get(meta.get('I', ''), 'Diğer / Belirsiz'),
                    'quire': meta.get('Q', ''),
                    'plant_id': '',
                    'comments': [],
                    'lines': [],
                }
                corpus[fid] = current
                continue
            if current is None:
                continue
            if line.startswith('#'):
                text = line.lstrip('# ').strip()
                if text:
                    current['comments'].append(text)
                    if text.startswith('Plant ID:'):
                        current['plant_id'] = text[len('Plant ID:'):].strip()
                continue
            lm = _LINE_TAG.match(line)
            if lm and lm.group(1) == current['folio']:
                locus = lm.group(3)
                toks = clean_line(lm.group(4))
                current['lines'].append({
                    'n': int(lm.group(2)),
                    'locus': locus,
                    'tokens': toks,
                    'raw': lm.group(4),
                    'paragraph_start': '<%>' in lm.group(4),
                })
    return corpus


def folio_tokens(folio_id: str, path: str = DEFAULT_TRANSCRIPT) -> List[str]:
    fid = folio_id.lower().lstrip('f') if folio_id.lower().startswith('f') else folio_id.lower()
    zl = load_zl(path)
    rec = zl.get(fid)
    return [t for l in rec['lines'] for t in l['tokens']] if rec else []


def normalize_folio_id(folio_id: str) -> str:
    fid = folio_id.strip().lower()
    return fid[1:] if fid.startswith('f') and fid[1:2].isdigit() else fid


def page_lines(folio_id: str, path: str = DEFAULT_TRANSCRIPT) -> List[Dict[str, Any]]:
    rec = load_zl(path).get(normalize_folio_id(folio_id))
    return rec['lines'] if rec else []


def tokens_in_scope(scope: str, folio_id: Optional[str] = None, path: str = DEFAULT_TRANSCRIPT) -> List[List[str]]:
    """Return list of lines (each a token list) for a scope.
    scope: 'page' | 'section' | 'language_a' | 'language_b' | 'all'
    """
    zl = load_zl(path)
    out: List[List[str]] = []
    cur = zl.get(normalize_folio_id(folio_id)) if folio_id else None
    for fid, rec in zl.items():
        if scope == 'page' and (cur is None or fid != cur['folio']):
            continue
        if scope == 'section' and (cur is None or rec['illustration'] != cur['illustration']):
            continue
        if scope == 'language_a' and rec['language'] != 'A':
            continue
        if scope == 'language_b' and rec['language'] != 'B':
            continue
        for l in rec['lines']:
            if l['tokens']:
                out.append(l['tokens'])
    return out
