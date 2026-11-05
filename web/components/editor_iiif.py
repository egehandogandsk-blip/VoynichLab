"""
VoynichLab Interactive Editor & OpenSeadragon IIIF Deep-Zoom Component
----------------------------------------------------------------------
1. CodeMirror 6 / Monaco style Transcription Editor with syntax highlighting
   for EVA, Currier, v101, Takahashi, and smart Voynich autocompletion.
2. OpenSeadragon IIIF Deep-Zoom Viewer with Annotorious vector overlay:
   - Box Tool (B key) for click-drag rectangle bounding boxes
   - Polygon Tool (P key) for curved text between drawings
   - Select & Pan Mode (S key / Escape)
   - 8-direction transform handles (resize, drag-and-drop, delete)
   - Inspect & Edit Popover for EVA transcription, morphological slot tag, and confidence
   - W3C Web Annotation JSON export
"""

import os
import json
import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit.components.v1 as components


def render_transcription_editor(initial_text: str, mode: str = "EVA", height: int = 340) -> None:
    """
    Renders a high-performance CodeMirror-style transcription editor with
    custom syntax highlighting and autocompletion for Voynich transcription systems.
    """
    escaped_text = json.dumps(initial_text)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.13/codemirror.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.13/theme/material-darker.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.13/addon/hint/show-hint.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.13/codemirror.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.13/addon/hint/show-hint.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #121212;
                color: #e5e5e5;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
            }}
            .editor-toolbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #181818;
                border: 1px solid #282828;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 6px 12px;
                font-size: 11px;
                color: #888;
            }}
            .editor-badge {{
                background: #242424;
                border: 1px solid #333;
                color: #ddd;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: 600;
            }}
            .CodeMirror {{
                height: {height - 40}px;
                background: #141414;
                border: 1px solid #282828;
                border-radius: 0 0 4px 4px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 13px;
                line-height: 1.5;
            }}
            /* Voynich syntax coloring */
            .cm-s-material-darker .cm-gallows {{ color: #38bdf8; font-weight: bold; }}
            .cm-s-material-darker .cm-bench {{ color: #818cf8; font-weight: bold; }}
            .cm-s-material-darker .cm-ligature {{ color: #fbbf24; }}
            .cm-s-material-darker .cm-vowel {{ color: #34d399; }}
            .cm-s-material-darker .cm-inflection {{ color: #f472b6; }}
            .cm-s-material-darker .cm-punctuation {{ color: #737373; }}
        </style>
    </head>
    <body>
        <div class="editor-toolbar">
            <div>
                <span>SİSTEM: </span><span class="editor-badge">{mode}</span>
                <span style="margin-left: 12px;">SÖZLÜK: </span><span class="editor-badge">EXTENDED EVA AUTOCOMPLETE</span>
            </div>
            <div>[Ctrl+Space: Otomatik Tamamlama]</div>
        </div>
        <textarea id="voynich-editor"></textarea>

        <script>
            // Register custom Voynich Mode
            CodeMirror.defineMode("voynich", function() {{
                const gallows = new Set(['k', 't', 'p', 'f']);
                const vowels = new Set(['o', 'a', 'y', 'e', 'i']);
                const benched = ['ckh', 'cth', 'cph', 'cfh'];
                const ligatures = ['ch', 'sh', 'ee'];

                return {{
                    token: function(stream) {{
                        if (stream.eatSpace()) return null;
                        
                        // Comments or tags
                        if (stream.match(/^<[^>]*>/)) return "comment";
                        if (stream.match(/^#.*/)) return "comment";

                        // Punctuation
                        if (stream.match(/^[.,\\-]/)) return "punctuation";

                        // Benched gallows
                        for (let b of benched) {{
                            if (stream.match(b)) return "bench";
                        }}
                        // Ligatures
                        for (let lig of ligatures) {{
                            if (stream.match(lig)) return "ligature";
                        }}
                        // Gallows
                        let ch = stream.next();
                        if (gallows.has(ch)) return "gallows";
                        if (vowels.has(ch)) return "vowel";
                        return null;
                    }}
                }};
            }});

            const voynichKeywords = [
                'daiin', 'qokedy', 'chedy', 'shedy', 'otchor', 'chol', 'chor',
                'qokain', 'cthaiin', 'dain', 'or', 'ar', 'al', 'ol', 'dal', 'dam',
                'otedy', 'okedy', 'chey', 'shey', 'qokeedy', 'sho', 'cho'
            ];

            CodeMirror.registerHelper("hint", "voynich", function(editor) {{
                const cur = editor.getCursor();
                const token = editor.getTokenAt(cur);
                const start = token.start;
                const end = cur.ch;
                const str = token.string.slice(0, end - start).toLowerCase();

                const list = voynichKeywords.filter(function(item) {{
                    return item.startsWith(str);
                }});

                return {{
                    list: list,
                    from: CodeMirror.Pos(cur.line, start),
                    to: CodeMirror.Pos(cur.line, end)
                }};
            }});

            const editor = CodeMirror.fromTextArea(document.getElementById("voynich-editor"), {{
                lineNumbers: true,
                mode: "voynich",
                theme: "material-darker",
                extraKeys: {{"Ctrl-Space": "autocomplete"}},
                lineWrapping: true
            }});

            editor.setValue({escaped_text});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)


def render_openseadragon_iiif_viewer(img_bgr: np.ndarray,
                                     boxes: List[Dict[str, Any]],
                                     height: int = 580,
                                     folio_id: str = "f11r") -> None:
    """
    Renders an OpenSeadragon Deep-Zoom viewer with Annotorious OpenSeadragon plugin:
    - Box Tool (B key) for click-drag rectangle bounding boxes
    - Polygon Tool (P key) for curved text between drawings
    - 8-direction Transform Handles for resize / drag
    - Popover Inspector for EVA transcription, word slot tags, and confidence rating
    - Pre-populated with detected word boxes as W3C Web Annotations
    """
    # Convert OpenCV image to base64 JPEG
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    H, W = img_bgr.shape[:2]

    # Convert detected boxes into standard W3C Web Annotations
    w3c_annotations = []
    for i, b in enumerate(boxes[:100]):
        x, y, w, h = b['bbox']
        conf = float(b.get('conf', 1.0))
        eva = b.get('eva') or f"w_{i+1}"
        slot = b.get('slot') or "core"
        w3c_annotations.append({
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": f"#anno-{i}",
            "type": "Annotation",
            "body": [
                {"type": "TextualBody", "purpose": "transcription", "value": str(eva)},
                {"type": "TextualBody", "purpose": "slot", "value": slot},
                {"type": "TextualBody", "purpose": "confidence", "value": str(round(conf * 100))}
            ],
            "target": {
                "selector": {
                    "type": "FragmentSelector",
                    "conformsTo": "http://www.w3.org/TR/media-frags/",
                    "value": f"xywh=pixel:{int(x)},{int(y)},{int(w)},{int(h)}"
                }
            }
        })
    annotations_json = json.dumps(w3c_annotations)

    beinecke_link = f"https://collections.library.yale.edu/catalog/2004245"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@recogito/annotorious-openseadragon@2.7.14/dist/annotorious.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@recogito/annotorious-openseadragon@2.7.14/dist/openseadragon-annotorious.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #121212;
                color: #e5e5e5;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                overflow: hidden;
            }}
            .viewer-toolbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #181818;
                border: 1px solid #282828;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 6px 12px;
                font-size: 11px;
                color: #888;
            }}
            .toolbar-group {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .viewer-container {{
                width: 100%;
                height: {height - 48}px;
                background: #0d0d0d;
                border: 1px solid #282828;
                border-radius: 0 0 4px 4px;
                position: relative;
            }}
            .tool-btn {{
                background: #242424;
                border: 1px solid #333;
                color: #ccc;
                padding: 3px 8px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 10px;
                font-family: 'JetBrains Mono', monospace;
                text-decoration: none;
                transition: all 0.15s ease;
            }}
            .tool-btn:hover {{ background: #333; color: #fff; border-color: #555; }}
            .tool-btn.active {{
                background: #0284c7;
                color: #fff;
                border-color: #38bdf8;
                font-weight: 600;
            }}
            .mode-indicator {{
                padding: 2px 6px;
                background: #1f1f1f;
                border: 1px solid #333;
                border-radius: 3px;
                font-size: 10px;
                color: #38bdf8;
            }}

            /* Annotorious Matte Black Customization */
            .a9s-annotationlayer .a9s-selection .a9s-inner,
            .a9s-annotationlayer .a9s-annotation .a9s-inner {{
                stroke: #38bdf8 !important;
                stroke-width: 1.5 !important;
                stroke-opacity: 0.9 !important;
                fill: rgba(56, 189, 248, 0.12) !important;
            }}
            .a9s-annotationlayer .a9s-annotation:hover .a9s-inner {{
                stroke: #f59e0b !important;
                fill: rgba(245, 158, 11, 0.22) !important;
            }}
            .a9s-annotationlayer .a9s-selection .a9s-handle .a9s-handle-inner {{
                fill: #38bdf8 !important;
                stroke: #ffffff !important;
                stroke-width: 1px !important;
            }}

            /* Floating Inspect & Edit Popover */
            #inspect-popover {{
                display: none;
                position: absolute;
                top: 48px;
                right: 14px;
                width: 280px;
                background: #181818;
                border: 1px solid #383838;
                border-radius: 6px;
                padding: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.7);
                z-index: 1000;
                font-size: 11px;
            }}
            #inspect-popover h4 {{
                margin: 0 0 8px 0;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #38bdf8;
                border-bottom: 1px solid #282828;
                padding-bottom: 4px;
            }}
            .pop-field {{
                margin-bottom: 8px;
            }}
            .pop-field label {{
                display: block;
                color: #888;
                margin-bottom: 3px;
                font-size: 10px;
            }}
            .pop-input, .pop-select {{
                width: 100%;
                box-sizing: border-box;
                background: #121212;
                border: 1px solid #333;
                color: #eee;
                padding: 4px 6px;
                border-radius: 3px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
            }}
            .pop-input:focus, .pop-select:focus {{
                border-color: #38bdf8;
                outline: none;
            }}
            .pop-actions {{
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
                border-top: 1px solid #282828;
                padding-top: 8px;
            }}
            .pop-btn {{
                padding: 3px 8px;
                font-size: 10px;
                border-radius: 3px;
                cursor: pointer;
                font-family: 'JetBrains Mono', monospace;
            }}
            .pop-btn-save {{ background: #0284c7; border: 1px solid #38bdf8; color: #fff; }}
            .pop-btn-del {{ background: #7f1d1d; border: 1px solid #ef4444; color: #fca5a5; }}
            .pop-btn-close {{ background: #262626; border: 1px solid #383838; color: #888; }}
        </style>
    </head>
    <body>
        <div class="viewer-toolbar">
            <div class="toolbar-group">
                <span style="color:#888;">VEKTÖR OSD: </span><span style="color:#fff; font-weight:600;">{folio_id}</span>
                <span class="mode-indicator" id="current-mode-tag">GEZİNME MODU</span>
                <span style="color:#666;">({W}x{H} px)</span>
            </div>
            <div class="toolbar-group">
                <button id="btn-select" class="tool-btn active" onclick="setMode('select')" title="Kısayol: S veya Esc">SEÇ / GEZİN (S)</button>
                <button id="btn-box" class="tool-btn" onclick="setMode('rect')" title="Kısayol: B">KUTU ARACI (B)</button>
                <button id="btn-poly" class="tool-btn" onclick="setMode('polygon')" title="Kısayol: P">ÇOKGEN ARACI (P)</button>
                <button class="tool-btn" onclick="exportAnnotations()">JSON DIŞA AKTAR</button>
                <button class="tool-btn" onclick="viewer.viewport.goHome()">SIFIRLA</button>
                <a class="tool-btn" href="{beinecke_link}" target="_blank">YALE ARŞİVİ ↗</a>
            </div>
        </div>

        <div id="osd-viewer" class="viewer-container">
            <!-- Popover Inspector -->
            <div id="inspect-popover">
                <h4>KELİME KUTUSU DENETÇİSİ</h4>
                <div class="pop-field">
                    <label>EVA Transkripsiyonu:</label>
                    <input type="text" id="pop-eva-input" class="pop-input" placeholder="ör. daiin, qokedy">
                </div>
                <div class="pop-field">
                    <label>Morfolojik Slot:</label>
                    <select id="pop-slot-select" class="pop-select">
                        <option value="prefix">Prefix (Ön Ek - qok, che, o)</option>
                        <option value="core" selected>Core (Gallows / Gövde - k, t, p, f, ch)</option>
                        <option value="suffix">Suffix (Son Ek - dy, in, y, aiin)</option>
                        <option value="uncertain">Belirsiz / Glif Grubu</option>
                    </select>
                </div>
                <div class="pop-field">
                    <label>Güven Skoru (1 - 5):</label>
                    <select id="pop-conf-select" class="pop-select">
                        <option value="5">5 - Kesin / Doğrulanmış (%100)</option>
                        <option value="4" selected>4 - Yüksek Güven (%80)</option>
                        <option value="3">3 - Orta Güven (%60)</option>
                        <option value="2">2 - Şüpheli / Düşük (%40)</option>
                        <option value="1">1 - Okunamayan / Hipotez (%20)</option>
                    </select>
                </div>
                <div class="pop-actions">
                    <button class="pop-btn pop-btn-del" onclick="deleteCurrentAnnotation()">SİL</button>
                    <div>
                        <button class="pop-btn pop-btn-close" onclick="closeInspector()">KAPAT</button>
                        <button class="pop-btn pop-btn-save" onclick="saveCurrentAnnotation()">KAYDET</button>
                    </div>
                </div>
                <button class="pop-btn" id="btn-analyze-inventory" onclick="analyzeAndAddToInventory()" style="background:#0284c7; color:#fff; border:1px solid #38bdf8; width:100%; margin-top:8px; padding:6px; font-weight:600; cursor:pointer;">
                    ⚡ SEÇİLİ ALANI ANALİZ ET & ENVANTERE EKLE
                </button>
                <div id="pop-notify" style="display:none; font-size:10px; color:#10b981; margin-top:6px; text-align:center;">
                    ✓ Alan analiz edildi ve envantere eklendi!
                </div>
            </div>
        </div>

        <script>
            let viewer, anno;
            let currentSelectedAnno = null;
            const initialAnnotations = {annotations_json};

            // Initialize OpenSeadragon
            viewer = OpenSeadragon({{
                id: "osd-viewer",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
                tileSources: {{
                    type: 'image',
                    url: 'data:image/jpeg;base64,{img_b64}'
                }},
                showNavigationControl: true,
                showNavigator: true,
                navigatorPosition: "BOTTOM_RIGHT",
                navigatorSizeRatio: 0.18,
                defaultZoomLevel: 1.0,
                minZoomLevel: 0.8,
                maxZoomLevel: 10.0,
                visibilityRatio: 1.0,
                constrainDuringPan: true
            }});

            // Initialize Annotorious OpenSeadragon plugin once viewer is ready
            viewer.addHandler('open', function() {{
                anno = OpenSeadragon.Annotorious(viewer, {{
                    allowEmpty: true,
                    disableEditor: true // We use our own customized matte black popover
                }});

                // Load initial bounding boxes as annotations
                if (initialAnnotations && initialAnnotations.length > 0) {{
                    initialAnnotations.forEach(function(a) {{
                        try {{
                            anno.addAnnotation(a);
                        }} catch (e) {{
                            console.warn("Annotation load skip:", e);
                        }}
                    }});
                }}

                // Event handlers
                anno.on('selectAnnotation', function(annotation) {{
                    openInspector(annotation);
                }});

                anno.on('createAnnotation', function(annotation) {{
                    openInspector(annotation);
                }});

                anno.on('cancelSelected', function() {{
                    closeInspector();
                }});
            }});

            // Mode switching
            function setMode(mode) {{
                if (!anno) return;
                document.querySelectorAll('.toolbar-group .tool-btn').forEach(b => b.classList.remove('active'));
                const tag = document.getElementById('current-mode-tag');

                if (mode === 'rect') {{
                    anno.setDrawingEnabled(true);
                    anno.setDrawingTool('rect');
                    document.getElementById('btn-box').classList.add('active');
                    tag.innerText = "KUTU ÇİZİMİ (B)";
                    tag.style.color = "#38bdf8";
                }} else if (mode === 'polygon') {{
                    anno.setDrawingEnabled(true);
                    anno.setDrawingTool('polygon');
                    document.getElementById('btn-poly').classList.add('active');
                    tag.innerText = "ÇOKGEN ÇİZİMİ (P)";
                    tag.style.color = "#f59e0b";
                }} else {{
                    anno.setDrawingEnabled(false);
                    document.getElementById('btn-select').classList.add('active');
                    tag.innerText = "GEZİNME MODU";
                    tag.style.color = "#888";
                }}
            }}

            // Keyboard shortcuts (B, P, S, Escape)
            document.addEventListener('keydown', function(e) {{
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
                const key = e.key.toLowerCase();
                if (key === 'b') {{
                    setMode('rect');
                }} else if (key === 'p') {{
                    setMode('polygon');
                }} else if (key === 's' || key === 'escape') {{
                    setMode('select');
                    closeInspector();
                }}
            }});

            // Inspector logic
            function openInspector(annotation) {{
                currentSelectedAnno = annotation;
                const pop = document.getElementById('inspect-popover');
                pop.style.display = 'block';

                let evaVal = "";
                let slotVal = "core";
                let confVal = "4";

                if (annotation.body && Array.isArray(annotation.body)) {{
                    annotation.body.forEach(function(b) {{
                        if (b.purpose === 'transcription') evaVal = b.value || "";
                        if (b.purpose === 'slot') slotVal = b.value || "core";
                        if (b.purpose === 'confidence') {{
                            const pct = parseInt(b.value, 10);
                            if (pct >= 90) confVal = "5";
                            else if (pct >= 75) confVal = "4";
                            else if (pct >= 50) confVal = "3";
                            else if (pct >= 30) confVal = "2";
                            else confVal = "1";
                        }}
                    }});
                }}

                document.getElementById('pop-eva-input').value = evaVal;
                document.getElementById('pop-slot-select').value = slotVal;
                document.getElementById('pop-conf-select').value = confVal;
            }}

            function closeInspector() {{
                document.getElementById('inspect-popover').style.display = 'none';
                currentSelectedAnno = null;
            }}

            function saveCurrentAnnotation() {{
                if (!currentSelectedAnno || !anno) return;
                const evaVal = document.getElementById('pop-eva-input').value.trim();
                const slotVal = document.getElementById('pop-slot-select').value;
                const confVal = document.getElementById('pop-conf-select').value;
                const confPct = confVal === "5" ? "100" : confVal === "4" ? "80" : confVal === "3" ? "60" : confVal === "2" ? "40" : "20";

                currentSelectedAnno.body = [
                    {{ "type": "TextualBody", "purpose": "transcription", "value": evaVal }},
                    {{ "type": "TextualBody", "purpose": "slot", "value": slotVal }},
                    {{ "type": "TextualBody", "purpose": "confidence", "value": confPct }}
                ];

                anno.updateSelected(currentSelectedAnno);
                closeInspector();
            }}

            function deleteCurrentAnnotation() {{
                if (!currentSelectedAnno || !anno) return;
                anno.removeAnnotation(currentSelectedAnno);
                closeInspector();
            }}

            function analyzeAndAddToInventory() {{
                if (!currentSelectedAnno) return;
                saveCurrentAnnotation();

                let evaVal = document.getElementById('pop-eva-input').value.trim() || 'w_custom';
                let slotVal = document.getElementById('pop-slot-select').value;
                let confVal = document.getElementById('pop-conf-select').value;

                let inventory = [];
                try {{
                    inventory = JSON.parse(localStorage.getItem('voynich_custom_inventory') || '[]');
                }} catch(e) {{}}

                inventory.push({{
                    id: currentSelectedAnno.id || ('anno_' + Date.now()),
                    folio: "{folio_id}",
                    eva: evaVal,
                    slot: slotVal,
                    conf: confVal,
                    target: currentSelectedAnno.target,
                    timestamp: Date.now()
                }});
                try {{
                    localStorage.setItem('voynich_custom_inventory', JSON.stringify(inventory));
                }} catch(e) {{}}

                const notify = document.getElementById('pop-notify');
                if (notify) {{
                    notify.innerText = `✓ '${{evaVal}}' analiz edildi ve envantere eklendi!`;
                    notify.style.display = 'block';
                    setTimeout(() => {{
                        notify.style.display = 'none';
                        closeInspector();
                    }}, 1600);
                }}
            }}

            function exportAnnotations() {{
                if (!anno) return;
                const all = anno.getAnnotations();
                const jsonStr = JSON.stringify(all, null, 2);
                const blob = new Blob([jsonStr], {{ type: "application/json" }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "{folio_id}_annotations.json";
                a.click();
                URL.revokeObjectURL(url);
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
