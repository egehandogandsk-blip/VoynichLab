import os
import sys
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.pdf_corpus import (generate_folio_sequence, build_corpus, load_corpus,
                            corpus_summary, section_of)


def test_folio_sequence_skips_missing():
    seq = generate_folio_sequence(6, start_folio=11)
    assert seq == ['f11r', 'f11v', 'f13r', 'f13v', 'f14r', 'f14v']
    assert generate_folio_sequence(2, 58, 'v') == ['f58v', 'f65r']
    assert section_of(11) == 'Bitkiler (Herbal)'


def test_build_corpus_from_pdf(tmp_path):
    fitz = pytest.importorskip('fitz')
    imgs = [os.path.join(base_dir, 'data', 'raw_images', n)
            for n in ('f11r.png', 'f11v.png', 'f16r.png')]
    if not all(os.path.exists(i) for i in imgs):
        pytest.skip('sample images missing')
    pdf_path = str(tmp_path / 'mini.pdf')
    doc = fitz.open()
    cover = doc.new_page()
    cover.insert_text((72, 72), 'cover')
    for p in imgs:
        page = doc.new_page(width=727, height=362)
        page.insert_image(page.rect, filename=p)
    doc.save(pdf_path)
    doc.close()

    corpus = build_corpus(pdf_path, 'pytest_mini', start_folio=11, page_offset=1)
    assert [p['folio_id'] for p in corpus['pages']] == ['f11r', 'f11v', 'f13r']
    assert corpus['pages'][0]['transcription']['available'] is True
    assert corpus['pages'][0]['vision']['words'] > 0
    assert load_corpus('pytest_mini')['total_pdf_pages'] == 4
    assert corpus_summary(corpus)['pages'] == 3
