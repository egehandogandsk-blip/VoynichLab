"""
Voynich Cryptologist Studio & Real-Time Linguistic Laboratory Engine
--------------------------------------------------------------------
Provides interactive tools for cryptanalysts and linguists:
1. Real-time substitution cipher simulator (EVA glyph -> Plaintext letter).
2. Lexical scoring against Medieval Botanical/Medical Latin & Italian lexicons.
3. Information-theoretic readability and entropy metrics.
4. Word-level and page-level annotation persistence and hypothesis testing.
5. Global lexicon propagation across the entire Voynich manuscript corpus.
"""

import os
import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

from src.eva import tokenize_units
from src.zl_corpus import load_zl, normalize_folio_id, tokens_in_scope
from src.crypto_entropy import compute_cross_entropy_and_kl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'data', 'annotations')
LEXICON_FILE = os.path.join(BASE_DIR, 'data', 'lexicon.json')
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Curated Medieval Botanical & Pharmacological Lexicon (Latin & Early Italian roots)
MEDIEVAL_BOTANICAL_LEXICON = {
    # Plant anatomy
    "radix", "radices", "radicibus", "folium", "folia", "foliis", "herba", "herbae", "herbam",
    "flos", "flores", "floribus", "fructus", "semen", "semina", "caulis", "rami", "ramus",
    "cortex", "ficus", "succus", "arbor", "planta", "spina", "foliolis", "stipula",
    # Medicinal qualities / Galenic humor
    "calida", "calidus", "calidum", "frigida", "frigidus", "frigidum", "sicca", "siccus", "siccum",
    "humida", "humidus", "humidum", "gradu", "primo", "secundo", "tertio", "quarto",
    # Pharmacological actions & preparations
    "aqua", "aquam", "oleum", "unguentum", "pulvis", "decoctio", "syrupus", "potio", "pilula",
    "coque", "cola", "terere", "bibe", "cape", "misce", "adhibe", "valet", "prodest", "contra",
    "dolorem", "febrim", "stomacho", "capiti", "oculis", "pectoris", "vulnera", "morsus",
    # Descriptions
    "magnus", "magna", "parvus", "parva", "albus", "alba", "niger", "nigra", "rubeus", "rubea",
    "viridis", "subtilis", "amara", "dulcis", "acuta", "bonus", "bona", "multum",
    # Early Italian / Venetian dialectal forms
    "radice", "foglia", "foglie", "erba", "fiore", "fiori", "seme", "rami", "pianta",
    "aqua", "oglio", "caldo", "freddo", "secco", "humido", "bono", "male", "dolore"
}

# Standard EVA glyph inventory
EVA_ALPHABET = [
    'a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'x', 'y'
]

# Curated Multi-Language Categorized Presets (Ancient, Hieroglyphs, 10 Modern Languages)
CATEGORIZED_CIPHER_PRESETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "Antik Diller & Alfabeler": {
        "Klasik Latince (Roma Fonetik)": {
            'o': 'o', 'k': 'c', 'e': 'e', 'd': 't', 'y': 'm',
            'a': 'a', 'i': 'i', 'n': 's', 'l': 'l', 'r': 'r',
            's': 'd', 't': 't', 'p': 'p', 'f': 'f', 'q': 'qu',
            'c': 'c', 'h': 'h', 'g': 'g', 'm': 'n', 'x': 'x'
        },
        "Ortaçağ Tıp Latincesi (15. yy Botanik)": {
            'o': 'o', 'k': 'c', 'e': 'e', 'd': 'x', 'y': 'm',
            'a': 'a', 'i': 'i', 'n': 's', 'l': 'l', 'r': 'r',
            's': 't', 't': 't', 'p': 'p', 'f': 'f', 'q': 'c',
            'c': 'ch', 'h': 'h', 'g': 'g', 'm': 'm', 'x': 'z'
        },
        "Klasik Grekçe (Bizans / İskenderiye)": {
            'o': 'ο', 'k': 'κ', 'e': 'ε', 'd': 'δ', 'y': 'υ',
            'a': 'α', 'i': 'ι', 'n': 'ν', 'l': 'λ', 'r': 'ρ',
            's': 'σ', 't': 'τ', 'p': 'π', 'f': 'φ', 'q': 'θ',
            'c': 'χ', 'h': 'η', 'g': 'γ', 'm': 'μ', 'x': 'ξ'
        },
        "İncil İbranicesi (Masoretik)": {
            'o': 'ו', 'k': 'כ', 'e': 'ה', 'd': 'ד', 'y': 'י',
            'a': 'א', 'i': 'י', 'n': 'נ', 'l': 'ל', 'r': 'ר',
            's': 'ס', 't': 'ת', 'p': 'פ', 'f': 'פ', 'q': 'ק',
            'c': 'כ', 'h': 'ח', 'g': 'ג', 'm': 'מ', 'x': 'צ'
        },
        "Süryanice (Syriac / Estrangela)": {
            'o': 'ܘ', 'k': 'ܟ', 'e': 'ܗ', 'd': 'ܕ', 'y': 'ܝ',
            'a': 'ܐ', 'i': 'ܝ', 'n': 'ܢ', 'l': 'ܠ', 'r': 'ܪ',
            's': 'ܣ', 't': 'ܬ', 'p': 'ܦ', 'f': 'ܦ', 'q': 'ܩ',
            'c': 'ܟ', 'h': 'ܚ', 'g': 'ܓ', 'm': 'ܡ', 'x': 'ܨ'
        },
        "Sanskritçe (Devanagari Fonetik)": {
            'o': 'ओ', 'k': 'क', 'e': 'ए', 'd': 'द', 'y': 'य',
            'a': 'अ', 'i': 'इ', 'n': 'न', 'l': 'ल', 'r': 'र',
            's': 'स', 't': 'त', 'p': 'प', 'f': 'फ', 'q': 'ख',
            'c': 'च', 'h': 'ह', 'g': 'ग', 'm': 'म', 'x': 'क्ष'
        },
        "Eski Kilise Slavcası (Kiril / Glagolitik)": {
            'o': 'о', 'k': 'к', 'e': 'е', 'd': 'д', 'y': 'и',
            'a': 'а', 'i': 'і', 'n': 'н', 'l': 'л', 'r': 'р',
            's': 'с', 't': 'т', 'p': 'п', 'f': 'ф', 'q': 'х',
            'c': 'ц', 'h': 'ч', 'g': 'г', 'm': 'м', 'x': 'ж'
        },
        "Gotça (Wulfila İncili)": {
            'o': '𐍉', 'k': '𐌺', 'e': '𐌴', 'd': '𐌳', 'y': '𐌹',
            'a': '𐌰', 'i': '𐌴𐌹', 'n': '𐌽', 'l': '𐌻', 'r': '𐍂',
            's': '𐍃', 't': '𐍄', 'p': '𐍀', 'f': '𐍆', 'q': '𐌵',
            'c': '𐌷', 'h': '𐌷', 'g': '𐌲', 'm': '𐌼', 'x': '𐌶'
        },
        "Aramice (Klasik İmparatorluk)": {
            'o': '𐡅', 'k': '𐡊', 'e': '𐡄', 'd': '𐡃', 'y': '𐡉',
            'a': '𐡀', 'i': '𐡉', 'n': '𐡍', 'l': '𐡋', 'r': '𐡓',
            's': '𐡎', 't': '𐡕', 'p': '𐡐', 'f': '𐡐', 'q': '𐡒',
            'c': '𐡊', 'h': '𐡇', 'g': '𐡂', 'm': '𐡌', 'x': '𐡑'
        },
        "Klasik Arapça (Endülüs / Tıp Külliyatı)": {
            'o': 'و', 'k': 'ك', 'e': 'ه', 'd': 'د', 'y': 'ي',
            'a': 'ا', 'i': 'ي', 'n': 'ن', 'l': 'ل', 'r': 'ر',
            's': 'س', 't': 'ت', 'p': 'ب', 'f': 'ف', 'q': 'ق',
            'c': 'خ', 'h': 'ح', 'g': 'ج', 'm': 'م', 'x': 'ص'
        }
    },
    "Hiyeroglifler & Çivi Yazısı": {
        "Mısır Hiyeroglifleri (Gardiner Fonogramları)": {
            'o': '𓅱', 'k': '𓎡', 'e': '𓇋', 'd': '𓂧', 'y': '𓇋𓇋',
            'a': '𓄿', 'i': '𓇋', 'n': '𓈖', 'l': '𓃭', 'r': '𓂋',
            's': '𓋴', 't': '𓏏', 'p': '𓊪', 'f': '𓆑', 'q': '𓏘',
            'c': '𓐍', 'h': '𓉔', 'g': '𓎼', 'm': '𓅓', 'x': '𓈎'
        },
        "Mezopotamya Çivi Yazısı (Akadca / Sümerce)": {
            'o': '𒌋', 'k': '𒅗', 'e': '𒂊', 'd': '𒁕', 'y': '𒅀',
            'a': '𒀀', 'i': '𒄿', 'n': '𒈾', 'l': '𒆷', 'r': '𒊏',
            's': '𒊓', 't': '𒋫', 'p': '𒉺', 'f': '𒉿', 'q': '𒋡',
            'c': '𒄩', 'h': '𒄭', 'g': '𒂵', 'm': '𒈠', 'x': '𒍝'
        },
        "Fenike Alfabesi (Proto-Kenan)": {
            'o': '𐤅', 'k': '𐤊', 'e': '𐤄', 'd': '𐤃', 'y': '𐤉',
            'a': '𐤀', 'i': '𐤉', 'n': '𐤍', 'l': '𐤋', 'r': '𐤓',
            's': '𐤎', 't': '𐤕', 'p': '𐤐', 'f': '𐤐', 'q': '𐤒',
            'c': '𐤇', 'h': '𐤄', 'g': '𐤂', 'm': '𐤌', 'x': '𐤑'
        },
        "Rünik Alfabe (Elder Futhark)": {
            'o': 'ᛟ', 'k': 'ᚲ', 'e': 'ᛖ', 'd': 'ᛞ', 'y': 'ᛃ',
            'a': 'ᚨ', 'i': 'ᛁ', 'n': 'ᚾ', 'l': 'ᛚ', 'r': 'ᚱ',
            's': 'ᛋ', 't': 'ᛏ', 'p': 'ᛈ', 'f': 'ᚠ', 'q': 'ᚲ',
            'c': 'ᚲ', 'h': 'ᚺ', 'g': 'ᚷ', 'm': 'ᛗ', 'x': 'ᛉ'
        },
        "Etrüsk Alfabesi (Arkaik İtalik)": {
            'o': '𐌖', 'k': '𐌊', 'e': '𐌄', 'd': '𐌕', 'y': '𐌉',
            'a': '𐌀', 'i': '𐌉', 'n': '𐌍', 'l': '𐌋', 'r': '𐌓',
            's': '𐌔', 't': '𐌕', 'p': '𐌐', 'f': '𐌅', 'q': '𐌒',
            'c': '𐌂', 'h': '7', 'g': '𐌂', 'm': '𐌌', 'x': '𐌘'
        },
        "Linear B (Miken Grekçesi Hece Harfleri)": {
            'o': '𐀃', 'k': '𐀏', 'e': '𐀁', 'd': '𐀅', 'y': '𐀊',
            'a': '𐀀', 'i': '𐀂', 'n': '𐀙', 'l': '𐀨', 'r': '𐀫',
            's': '𐀭', 't': '𐀲', 'p': '𐀞', 'f': '𐀡', 'q': '𐀦',
            'c': '𐀐', 'h': '𐀂', 'g': '𐀐', 'm': '𐀔', 'x': '𐀼'
        }
    },
    "En Yaygın 10 Modern Dil": {
        "İngilizce (Modern English)": {
            'o': 'e', 'y': 't', 'a': 'a', 'd': 'o', 'e': 'i',
            'i': 'n', 'n': 's', 'l': 'h', 'k': 'r', 'h': 'd',
            'r': 'l', 's': 'c', 't': 'u', 'p': 'm', 'f': 'w',
            'q': 'f', 'c': 'g', 'g': 'y', 'm': 'p', 'x': 'b'
        },
        "İspanyolca (Modern Español)": {
            'o': 'e', 'y': 'a', 'a': 'o', 'd': 's', 'e': 'i',
            'i': 'n', 'n': 'r', 'l': 'l', 'k': 'd', 'h': 'c',
            'r': 't', 's': 'u', 't': 'm', 'p': 'p', 'f': 'b',
            'q': 'q', 'c': 'g', 'g': 'v', 'm': 'y', 'x': 'j'
        },
        "Fransızca (Français)": {
            'o': 'e', 'y': 's', 'a': 'a', 'd': 'i', 'e': 't',
            'i': 'n', 'n': 'r', 'l': 'u', 'k': 'l', 'h': 'o',
            'r': 'd', 's': 'c', 't': 'p', 'p': 'm', 'f': 'v',
            'q': 'q', 'c': 'g', 'g': 'f', 'm': 'b', 'x': 'j'
        },
        "Almanca (Deutsch)": {
            'o': 'e', 'y': 'n', 'a': 'i', 'd': 's', 'e': 'r',
            'i': 'a', 'n': 't', 'l': 'd', 'k': 'h', 'h': 'u',
            'r': 'l', 's': 'c', 't': 'g', 'p': 'm', 'f': 'o',
            'q': 'b', 'c': 'w', 'g': 'f', 'm': 'k', 'x': 'z'
        },
        "İtalyanca (Italiano)": {
            'o': 'e', 'y': 'a', 'a': 'i', 'd': 'o', 'e': 'n',
            'i': 'r', 'n': 't', 'l': 'l', 'k': 's', 'h': 'c',
            'r': 'd', 's': 'u', 't': 'm', 'p': 'p', 'f': 'v',
            'q': 'g', 'c': 'b', 'g': 'f', 'm': 'z', 'x': 'q'
        },
        "Portekizce (Português)": {
            'o': 'a', 'y': 'e', 'a': 'o', 'd': 's', 'e': 'r',
            'i': 'i', 'n': 'n', 'l': 'd', 'k': 'm', 'h': 't',
            'r': 'u', 's': 'c', 't': 'l', 'p': 'p', 'f': 'v',
            'q': 'g', 'c': 'q', 'g': 'b', 'm': 'f', 'x': 'z'
        },
        "Rusça (Modern Kiril)": {
            'o': 'о', 'y': 'е', 'a': 'а', 'd': 'и', 'e': 'н',
            'i': 'т', 'n': 'с', 'l': 'р', 'k': 'в', 'h': 'л',
            'r': 'к', 's': 'м', 't': 'д', 'p': 'п', 'f': 'у',
            'q': 'я', 'c': 'ы', 'g': 'ь', 'm': 'г', 'x': 'з'
        },
        "Türkçe (Latin Türk Alfabesi)": {
            'o': 'a', 'y': 'e', 'a': 'i', 'd': 'n', 'e': 'r',
            'i': 'l', 'n': 'ı', 'l': 'd', 'k': 'k', 'h': 'm',
            'r': 'u', 's': 't', 't': 's', 'p': 'b', 'f': 'y',
            'q': 'o', 'c': 'ü', 'g': 'ş', 'm': 'z', 'x': 'g'
        },
        "Çince (Pinyin Harf Frekansı)": {
            'o': 'i', 'y': 'a', 'a': 'n', 'd': 'g', 'e': 'e',
            'i': 'u', 'n': 'o', 'l': 'h', 'k': 'z', 'h': 's',
            'r': 'd', 's': 't', 't': 'b', 'p': 'l', 'f': 'm',
            'q': 'j', 'c': 'x', 'g': 'q', 'm': 'c', 'x': 'p'
        },
        "Standart Modern Arapça (Fusha)": {
            'o': 'ا', 'y': 'ل', 'a': 'م', 'd': 'ي', 'e': 'و',
            'i': 'ن', 'n': 'ر', 'l': 'ب', 'k': 'ت', 'h': 'ه',
            'r': 'ع', 's': 'ف', 't': 'د', 'p': 'س', 'f': 'ق',
            'q': 'ك', 'c': 'ح', 'g': 'ص', 'm': 'ط', 'x': 'خ'
        }
    }
}

# Flatten for backward compatibility
CIPHER_PRESETS: Dict[str, Dict[str, str]] = {
    "Sıfırla (Birebir EVA)": {g: g for g in EVA_ALPHABET}
}
for cat, presets in CATEGORIZED_CIPHER_PRESETS.items():
    for name, mapping in presets.items():
        CIPHER_PRESETS[f"[{cat}] {name}"] = mapping
        CIPHER_PRESETS[name] = mapping

# Legacy alias keys with defensive lookup
antik = CATEGORIZED_CIPHER_PRESETS.get("Antik Diller & Alfabeler", {})
modern = CATEGORIZED_CIPHER_PRESETS.get("En Yaygın 10 Modern Dil", {})

for k, v in antik.items():
    if "Latince" in k and "Roma" in k:
        CIPHER_PRESETS["Latince Fonetik Hipotezi"] = v
    elif "Medikal" in k or "Tıp" in k or "Ortaçağ" in k:
        CIPHER_PRESETS["İtalyanca / Venedik Tıp Hipotezi"] = v
    elif "Grek" in k:
        CIPHER_PRESETS["Grekçe Fonetik Hipotezi"] = v
    elif "İbrani" in k or "Ibrani" in k:
        CIPHER_PRESETS["İbranice Hipotezi"] = v

for k, v in modern.items():
    if "İtalyanca" in k or "Italyanca" in k:
        CIPHER_PRESETS["Frekans Dağılım Eşleşmesi (Voynich -> Latin)"] = v




class SubstitutionCipher:
    """Manages substitution cipher mappings and real-time plaintext evaluation."""

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping: Dict[str, str] = {g: g for g in EVA_ALPHABET}
        if mapping:
            self.mapping.update(mapping)

    def set_mapping(self, glyph: str, target: str) -> None:
        self.mapping[glyph.lower()] = target.lower()

    def decipher_token(self, token: str) -> str:
        """Applies character substitutions while respecting multigraphs."""
        out: List[str] = []
        units = tokenize_units(token)
        for u in units:
            out.append(self.mapping.get(u, u))
        return "".join(out)

    def decipher_tokens(self, tokens: List[str]) -> List[str]:
        return [self.decipher_token(t) for t in tokens]

    def evaluate_decipherment(self, tokens: List[str], target_lang: str = 'latin') -> Dict[str, Any]:
        """
        Evaluates a set of deciphered tokens:
        - Lexicon hits against medieval botanical/medical Latin & Italian
        - Shannon conditional entropy H2
        - Vowel-to-consonant ratio
        - Plausibility score (0.0 to 1.0)
        """
        if not tokens:
            return {
                'total_words': 0, 'lexicon_hits': 0, 'hit_rate': 0.0,
                'matched_words': [], 'entropy_h2': 0.0, 'vowel_ratio': 0.0,
                'plausibility_score': 0.0
            }

        deciphered = self.decipher_tokens(tokens)
        text = " ".join(deciphered)

        # Lexicon hits
        hits = []
        for dec, orig in zip(deciphered, tokens):
            cleaned = dec.strip().lower()
            if cleaned in MEDIEVAL_BOTANICAL_LEXICON:
                hits.append({'original_eva': orig, 'deciphered': cleaned, 'meaning': 'Latince/İtalyanca Tıbbi Terim'})

        hit_rate = len(hits) / max(len(tokens), 1)

        # Conditional bigram entropy H2
        chars = [c for c in text.lower() if c.isalpha()]
        h2 = 0.0
        if len(chars) > 2:
            bigrams = Counter(zip(chars[:-1], chars[1:]))
            unigrams = Counter(chars[:-1])
            tot = sum(bigrams.values())
            for (c1, c2), count in bigrams.items():
                p_xy = count / tot
                p_y_given_x = count / unigrams[c1]
                h2 -= p_xy * math.log2(p_y_given_x)

        # Vowel ratio
        vowels = set('aeiou')
        v_count = sum(1 for c in chars if c in vowels)
        v_ratio = v_count / max(len(chars), 1)

        # Plausibility score combining dictionary matches and phonetic distribution
        v_penalty = abs(v_ratio - 0.44) * 2.0
        h_score = min(h2 / 4.0, 1.0)
        plausibility = min(1.0, max(0.0, (hit_rate * 4.0) + (0.3 * h_score) + max(0, 0.3 - v_penalty)))

        # Cross-entropy against target language
        ce_kl = compute_cross_entropy_and_kl(deciphered, target_lang=target_lang)

        return {
            'total_words': len(tokens),
            'lexicon_hits': len(hits),
            'hit_rate': round(hit_rate * 100, 2),
            'matched_words': hits[:15],
            'entropy_h2': round(h2, 2),
            'vowel_ratio': round(v_ratio * 100, 1),
            'plausibility_score': round(plausibility * 100, 1),
            'cross_entropy': ce_kl['cross_entropy'],
            'kl_divergence': ce_kl['kl_divergence']
        }



class PageAnnotationManager:
    """Manages user edits, custom readings, translations and POS tags per folio."""

    def __init__(self, folio_id: str):
        self.folio_id = normalize_folio_id(folio_id)
        self.filepath = os.path.join(ANNOTATIONS_DIR, f"{self.folio_id}.json")
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'folio': self.folio_id,
            'word_annotations': {},  # str(word_index) -> dict
            'token_annotations': {}, # raw_eva_token -> dict
            'folio_notes': '',
            'hypothesis_author': 'Kriptolog'
        }

    def save(self) -> None:
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def set_word_annotation(self, word_idx: int, reading: str, meaning: str,
                            pos: str, slot: Optional[Dict[str, str]] = None,
                            notes: str = '', confidence: int = 3) -> None:
        self.data['word_annotations'][str(word_idx)] = {
            'reading': reading,
            'meaning': meaning,
            'pos': pos,
            'slot': slot or {},
            'notes': notes,
            'confidence': confidence
        }
        if reading:
            self.data['token_annotations'][reading] = {
                'meaning': meaning,
                'pos': pos,
                'notes': notes
            }
        self.save()

    def get_word_annotation(self, word_idx: int) -> Optional[Dict[str, Any]]:
        return self.data['word_annotations'].get(str(word_idx))

    def get_token_annotation(self, token: str) -> Optional[Dict[str, Any]]:
        return self.data['token_annotations'].get(token)


class GlobalLexicon:
    """Central repository of user-discovered vocabulary and translations across all folios."""

    def __init__(self, filepath: str = LEXICON_FILE):
        self.filepath = filepath
        self.entries: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save(self) -> None:
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add_entry(self, token_eva: str, meaning: str, pos: str,
                  source_folio: str = '', notes: str = '') -> None:
        token = token_eva.strip().lower()
        if not token:
            return
        if token not in self.entries:
            self.entries[token] = {
                'meaning': meaning,
                'pos': pos,
                'folios': [source_folio] if source_folio else [],
                'notes': notes,
                'verified': False
            }
        else:
            self.entries[token]['meaning'] = meaning
            self.entries[token]['pos'] = pos
            if source_folio and source_folio not in self.entries[token]['folios']:
                self.entries[token]['folios'].append(source_folio)
            if notes:
                self.entries[token]['notes'] = notes
        self.save()

    def lookup(self, token_eva: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(token_eva.strip().lower())

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return self.entries
