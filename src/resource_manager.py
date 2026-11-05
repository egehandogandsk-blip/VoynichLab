"""
VoynichLab Resource Manager
Handles categorization, metadata extraction, renaming, replacing,
and deleting of single-page folio images and multi-page PDF corpora.
"""

import os
import re
import json
import shutil
from typing import Dict, List, Any, Tuple
import cv2


def get_file_size_str(num_bytes: int) -> str:
    """Return human-readable file size."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------- Single Image Folio Assets
def list_image_assets(raw_img_dir: str) -> List[Dict[str, Any]]:
    """Lists all image files in raw_images directory with metadata."""
    if not os.path.isdir(raw_img_dir):
        return []

    assets = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for fname in sorted(os.listdir(raw_img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue
        full_path = os.path.join(raw_img_dir, fname)
        if not os.path.isfile(full_path):
            continue

        size_bytes = os.path.getsize(full_path)
        folio_id = os.path.splitext(fname)[0]

        w, h = 0, 0
        try:
            im = cv2.imread(full_path)
            if im is not None:
                h, w = im.shape[:2]
        except Exception:
            pass

        is_catalog = folio_id in {"f11r", "f11v", "f16r", "f18r"}

        assets.append({
            "filename": fname,
            "folio_id": folio_id,
            "path": full_path,
            "size_bytes": size_bytes,
            "size_str": get_file_size_str(size_bytes),
            "width": w,
            "height": h,
            "is_catalog": is_catalog,
        })
    return assets


def rename_image_asset(old_filename: str, new_name: str, raw_img_dir: str) -> Tuple[bool, str, str]:
    """Rename an image asset. new_name can be a bare folio name without extension."""
    old_path = os.path.join(raw_img_dir, old_filename)
    if not os.path.exists(old_path):
        return False, f"Dosya bulunamadı: {old_filename}", old_filename

    ext = os.path.splitext(old_filename)[1].lower()
    clean_name = re.sub(r'[^A-Za-z0-9_-]+', '_', new_name.strip()).strip('_')
    if not clean_name:
        return False, "Geçersiz dosya adı.", old_filename

    new_filename = f"{clean_name}{ext}" if not clean_name.endswith(ext) else clean_name
    new_path = os.path.join(raw_img_dir, new_filename)

    if new_filename == old_filename:
        return True, "İsim değişmedi.", old_filename

    if os.path.exists(new_path):
        return False, f"Bu isimde bir dosya zaten var: {new_filename}", old_filename

    try:
        os.rename(old_path, new_path)
        return True, f"'{old_filename}' başarıyla '{new_filename}' olarak adlandırıldı.", new_filename
    except Exception as exc:
        return False, f"Adlandırma hatası: {str(exc)}", old_filename


def replace_image_asset(filename: str, file_bytes: bytes, raw_img_dir: str) -> Tuple[bool, str]:
    """Overwrites an existing image asset with new byte contents."""
    path = os.path.join(raw_img_dir, filename)
    try:
        with open(path, "wb") as f:
            f.write(file_bytes)
        return True, f"'{filename}' başarıyla güncellendi."
    except Exception as exc:
        return False, f"Değiştirme hatası: {str(exc)}"


def delete_image_asset(filename: str, raw_img_dir: str) -> Tuple[bool, str]:
    """Deletes an image asset from raw_images directory."""
    path = os.path.join(raw_img_dir, filename)
    if not os.path.exists(path):
        return False, f"Dosya bulunamadı: {filename}"
    try:
        os.remove(path)
        return True, f"'{filename}' başarıyla silindi."
    except Exception as exc:
        return False, f"Silme hatası: {str(exc)}"


# ---------------------------------------------------------------------- Multi-page PDF Corpora Assets
def list_corpus_assets(corpus_root: str, uploads_dir: str) -> List[Dict[str, Any]]:
    """Lists all stored PDF corpora with page and image generation status."""
    if not os.path.isdir(corpus_root):
        return []

    corpora = []
    for cname in sorted(os.listdir(corpus_root)):
        cdir = os.path.join(corpus_root, cname)
        cjson = os.path.join(cdir, "corpus.json")
        if not os.path.isfile(cjson):
            continue

        try:
            with open(cjson, "r", encoding="utf-8") as f:
                cdata = json.load(f)
        except Exception:
            cdata = {}

        pages = cdata.get("pages", [])
        source_pdf = cdata.get("source_pdf", f"{cname}.pdf")
        pdf_path = os.path.join(uploads_dir, source_pdf)
        pdf_exists = os.path.isfile(pdf_path)

        pages_dir = os.path.join(cdir, "pages")
        existing_pages = len(os.listdir(pages_dir)) if os.path.isdir(pages_dir) else 0

        total_pages = len(pages)
        pdf_size_bytes = os.path.getsize(pdf_path) if pdf_exists else 0

        corpora.append({
            "name": cname,
            "manifest_path": cjson,
            "total_pages": total_pages,
            "rendered_pages": existing_pages,
            "pages_complete": (existing_pages >= total_pages and total_pages > 0),
            "source_pdf": source_pdf,
            "pdf_exists": pdf_exists,
            "pdf_size_str": get_file_size_str(pdf_size_bytes) if pdf_exists else "-",
        })
    return corpora


def rename_corpus_asset(old_name: str, new_name: str, corpus_root: str) -> Tuple[bool, str, str]:
    """Rename a corpus directory and update internal corpus.json manifest."""
    old_dir = os.path.join(corpus_root, old_name)
    if not os.path.isdir(old_dir):
        return False, f"Külliyat bulunamadı: {old_name}", old_name

    clean_name = re.sub(r'[^A-Za-z0-9_-]+', '_', new_name.strip()).strip('_')
    if not clean_name:
        return False, "Geçersiz külliyat adı.", old_name

    new_dir = os.path.join(corpus_root, clean_name)
    if clean_name == old_name:
        return True, "İsim değişmedi.", old_name

    if os.path.exists(new_dir):
        return False, f"Bu isimde bir külliyat zaten var: {clean_name}", old_name

    try:
        cjson = os.path.join(old_dir, "corpus.json")
        if os.path.isfile(cjson):
            with open(cjson, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["name"] = clean_name
            with open(cjson, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)

        os.rename(old_dir, new_dir)
        return True, f"Külliyat '{new_name}' olarak güncellendi.", clean_name
    except Exception as exc:
        return False, f"Külliyat adlandırma hatası: {str(exc)}", old_name


def delete_corpus_asset(name: str, corpus_root: str, uploads_dir: str = "") -> Tuple[bool, str]:
    """Deletes a corpus directory and all its rendered page caches."""
    cdir = os.path.join(corpus_root, name)
    if not os.path.isdir(cdir):
        return False, f"Külliyat bulunamadı: {name}"

    try:
        shutil.rmtree(cdir)
        return True, f"'{name}' külliyatı ve tüm sayfaları başarıyla silindi."
    except Exception as exc:
        return False, f"Külliyat silme hatası: {str(exc)}"
