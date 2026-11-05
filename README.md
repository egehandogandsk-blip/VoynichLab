# Voynich Manuscript: Computational Linguistics & Cryptanalysis Lab

Bu sistem, Voynich El Yazması (Beinecke MS 408) üzerindeki bilinmeyen yazı ve dil sistemini çözmek üzere geliştirilmiş uçtan uca hesaplamalı dilbilim ve kriptoanaliz yazılım paketidir.

## 🚀 Hızlı Başlangıç

### 1. Testleri Çalıştırma
```bash
pytest tests/test_pipeline.py -v
```

### 2. İnteraktif Web Panelini Başlatma
```bash
streamlit run web/app.py
```

## 📂 Proje Yapısı

- `src/vision_segmenter.py`: Parşömen gürültüsünü filtreleme, metin/çizim ayrımı, satır ve kelime kutulama.
- `src/glyph_clustering.py`: Şekil öznitelik vektörleri (Hu momentleri, ızgara yoğunluğu) ile denetimsiz glif kümeleme (Unsupervised OCR).
- `src/slot_grammar.py`: Katı sonlu durum makinesi (Finite State Machine) ile kelime morfolojisi ayrıştırma: `[Önek] + [Medial/Gallows] + [Çekirdek Vokal] + [Sonek]`.
- `src/crypto_entropy.py`: Shannon 1. ve 2. derece şartlı entropisi ($H_1, H_2$), Index of Coincidence, Zipf Yasası güç uyumu, Boris Sukhotin vokal keşif algoritması ve Currier A/B lehçe sınıflandırıcısı.
- `src/crib_engine.py`: Ortaçağ botanik külliyatı (*Dioscorides, Herbarius Moguntinus*) ve izomorfik kelime tekrar deseni eşleyicisi.
- `src/pipeline.py`: Tüm katmanları birleştiren orkestrasyon motoru.
- `web/app.py`: Streamlit tabanlı interaktif görsel analiz paneli.
- `data/raw_images/`: Kullanıcının sağladığı sayfalar (`f11r.png`, `f11v.png`, `f16r.png`, `f18r.png`).
- `data/transcripts/`: ZL3b-n (Extended EVA) ve IT2a-n (Takahashi) transkripsiyon külliyatı.

- `src/pdf_corpus.py`: PDF'i sayfa sayfa g�rsele �evirir, her sayfaya folio kimli�i (f1r, f1v, ...) atar, analiz eder ve `data/corpus/<ad>/corpus.json` olarak kaydeder.
