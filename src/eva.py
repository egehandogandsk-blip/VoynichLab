"""
EVA helpers: glyph-unit tokenization and small utilities shared by all modules.
"""

import re
from typing import List

# Composite glyphs are matched first (longest first)
COMPOSITES = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']
_COMPOSITE_RE = re.compile('|'.join(COMPOSITES))

GALLOWS = set('ktpf')


def eva_units(token: str, composite: bool = True) -> List[str]:
    """Split an EVA token into glyph units. composite=True keeps ch/sh/cXh as single glyphs."""
    t = re.sub(r'[^a-z?]', '', token.lower())
    if not composite:
        return list(t)
    units: List[str] = []
    i = 0
    while i < len(t):
        m = _COMPOSITE_RE.match(t, i)
        if m:
            units.append(m.group(0))
            i = m.end()
        else:
            units.append(t[i])
            i += 1
    return units


def unit_len(token: str) -> int:
    return max(1, len(eva_units(token)))


tokenize_units = eva_units

