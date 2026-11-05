import os
import tempfile
import cv2
import numpy as np
import pytest

from src.resource_manager import (
    list_image_assets, rename_image_asset, replace_image_asset,
    delete_image_asset, list_corpus_assets, rename_corpus_asset, delete_corpus_asset
)
from src.pdf_corpus import ensure_page_image, load_corpus
from src.vision_segmenter import load_image


def test_resource_manager_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_p = os.path.join(tmpdir, 'f99r.png')
        cv2.imwrite(img_p, np.full((120, 80, 3), 200, dtype=np.uint8))
        
        assets = list_image_assets(tmpdir)
        assert len(assets) == 1
        assert assets[0]['folio_id'] == 'f99r'
        assert assets[0]['width'] == 80
        assert assets[0]['height'] == 120
        
        # Rename
        ok, msg, new_fn = rename_image_asset('f99r.png', 'f99_custom', tmpdir)
        assert ok
        assert new_fn == 'f99_custom.png'
        assert os.path.exists(os.path.join(tmpdir, 'f99_custom.png'))
        
        # Replace
        fake_bytes = b'TEST_RAW_DATA'
        ok, msg = replace_image_asset('f99_custom.png', fake_bytes, tmpdir)
        assert ok
        with open(os.path.join(tmpdir, 'f99_custom.png'), 'rb') as f:
            assert f.read() == fake_bytes
            
        # Delete
        ok, msg = delete_image_asset('f99_custom.png', tmpdir)
        assert ok
        assert not os.path.exists(os.path.join(tmpdir, 'f99_custom.png'))


def test_safe_load_image_fallback():
    # If file does not exist, load_image must NEVER crash and return valid canvas
    non_existent = os.path.join('data', 'raw_images', 'definitely_does_not_exist_xyz.png')
    img = load_image(non_existent)
    assert isinstance(img, np.ndarray)
    assert img.shape == (1000, 750, 3)


def test_ensure_page_image_resilience():
    page_entry = {'folio_id': 'test_folio_resilience', 'pdf_page': 1}
    ensured = ensure_page_image('test_corpus_tmp', page_entry)
    assert os.path.isfile(ensured)
    img = load_image(ensured)
    assert isinstance(img, np.ndarray)
    assert img.shape[0] > 0
    # Cleanup
    if os.path.exists(ensured):
        os.remove(ensured)
