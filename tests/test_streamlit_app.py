"""
Streamlit AppTest Verification
"""

import os
import sys
import pytest
from streamlit.testing.v1 import AppTest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_renders_without_exceptions():
    app_path = os.path.join(base_dir, "web", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest threw an exception: {at.exception}"


def test_app_renders_e2canvas_mode():
    app_path = os.path.join(base_dir, "web", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.session_state['e2canvas_mode'] = True
    at.run()
    assert not at.exception, f"AppTest e2canvas_mode threw an exception: {at.exception}"


def test_app_renders_openseadragon_mode():
    app_path = os.path.join(base_dir, "web", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    at.radio[0].set_value("OpenSeadragon IIIF Deep-Zoom Tuvali").run()
    assert not at.exception, f"AppTest openseadragon threw an exception: {at.exception}"


def test_app_renders_tabs_and_inventories():
    app_path = os.path.join(base_dir, "web", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    # Check that sidebar elements and main tabs exist
    assert len(at.tabs) >= 9

