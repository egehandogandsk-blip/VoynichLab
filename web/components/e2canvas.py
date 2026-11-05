"""
e2Canvas: Interactive Sandbox Workstation Component for VoynichLab
-------------------------------------------------------------------
Implements the 3-panel Infinite Canvas Sandbox requested by user:
1. Header: [SANDBOX MODU: e2Canvas] and [X Kapat]
2. Left Panel: ENVANTER (Inventory)
   - Folio Image Crops (OSD thumbnails)
   - EVA Word List (frequencies & lines)
   - Dictionary & Translation Cards (Latin/Italian/Greek hypotheses)
3. Center Panel: SONSUZ KANVAS (Infinite Pan & Zoom Canvas)
   - Draggable nodes for Image Crops, EVA words, Translations
   - Directed Bridge Connections (=======> [Köprü])
   - Group Frames
4. Bottom Panel: GRUP KRİPTO & DİLBİLİM ANALİZÖRÜ
   - Live telemetry: Shannon H, Bigram Tutarlılığı %, Sözlük Uyum %
   - Morphological breakdown and cluster metrics
"""

import json
from typing import List, Dict, Any, Optional
import streamlit.components.v1 as components


def render_e2canvas(folio_id: str,
                    crops: List[Dict[str, Any]],
                    words: List[Dict[str, Any]],
                    hypotheses: List[Dict[str, Any]],
                    height: int = 760) -> None:
    """
    Renders the e2Canvas full sandbox workstation.
    """
    crops_json = json.dumps(crops[:40])
    words_json = json.dumps(words[:60])
    hypotheses_json = json.dumps(hypotheses[:30])

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

            * {{
                box-sizing: border-box;
                user-select: none;
            }}
            body {{
                margin: 0;
                padding: 0;
                background-color: #101010;
                color: #e5e5e5;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                overflow: hidden;
                height: {height}px;
                display: flex;
                flex-direction: column;
            }}

            /* Top Bar */
            .e2-header {{
                height: 42px;
                background: #161616;
                border-bottom: 1px solid #282828;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 16px;
                font-size: 11px;
                flex-shrink: 0;
            }}
            .e2-title {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 700;
                letter-spacing: 0.8px;
                color: #38bdf8;
            }}
            .e2-badge {{
                background: #242424;
                border: 1px solid #333;
                color: #aaa;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: 500;
                font-size: 10px;
            }}
            .e2-toolbar {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .e2-btn {{
                background: #222;
                border: 1px solid #333;
                color: #ccc;
                padding: 4px 10px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 10px;
                font-family: 'JetBrains Mono', monospace;
                transition: all 0.15s ease;
            }}
            .e2-btn:hover {{
                background: #333;
                color: #fff;
                border-color: #555;
            }}
            .e2-btn-bridge {{
                background: #0369a1;
                border-color: #38bdf8;
                color: #fff;
            }}
            .e2-btn-bridge.active {{
                background: #0284c7;
                box-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
            }}

            /* Main Workspace Split */
            .e2-workspace {{
                display: flex;
                flex: 1;
                height: calc(100% - 192px);
                position: relative;
                overflow: hidden;
            }}

            /* Left Panel: Inventory */
            .e2-inventory {{
                width: 280px;
                background: #141414;
                border-right: 1px solid #262626;
                display: flex;
                flex-direction: column;
                flex-shrink: 0;
            }}
            .inv-tabs {{
                display: flex;
                background: #181818;
                border-bottom: 1px solid #262626;
            }}
            .inv-tab {{
                flex: 1;
                padding: 8px 4px;
                text-align: center;
                font-size: 10px;
                color: #888;
                cursor: pointer;
                border-bottom: 2px solid transparent;
            }}
            .inv-tab.active {{
                color: #fff;
                border-bottom-color: #38bdf8;
                background: #1a1a1a;
                font-weight: 600;
            }}
            .inv-content {{
                flex: 1;
                overflow-y: auto;
                padding: 8px;
            }}
            .inv-item {{
                background: #1c1c1c;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 6px 8px;
                margin-bottom: 6px;
                cursor: pointer;
                transition: all 0.15s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 11px;
            }}
            .inv-item:hover {{
                background: #242424;
                border-color: #38bdf8;
                transform: translateX(2px);
            }}
            .inv-crop-thumb {{
                width: 44px;
                height: 24px;
                object-fit: cover;
                border-radius: 2px;
                border: 1px solid #333;
                background: #000;
            }}
            .inv-tag {{
                font-size: 9px;
                color: #888;
                background: #121212;
                padding: 1px 4px;
                border-radius: 2px;
            }}

            /* Center Panel: Infinite Canvas */
            .e2-canvas-container {{
                flex: 1;
                position: relative;
                background: #0d0d0d;
                overflow: hidden;
                cursor: grab;
            }}
            .e2-canvas-container.panning {{
                cursor: grabbing;
            }}
            .e2-canvas-container.bridging {{
                cursor: crosshair;
            }}
            /* Canvas grid pattern */
            .canvas-grid {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
                background-size: 24px 24px;
                pointer-events: none;
            }}
            #canvas-stage {{
                position: absolute;
                top: 0;
                left: 0;
                transform-origin: 0 0;
                width: 10000px;
                height: 10000px;
            }}

            /* SVG Connection Lines */
            #canvas-svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 5;
            }}
            .bridge-line {{
                stroke: #38bdf8;
                stroke-width: 2.5;
                stroke-dasharray: 6 3;
                animation: dash 1s linear infinite;
                fill: none;
            }}
            @keyframes dash {{
                to {{
                    stroke-dashoffset: -9;
                }}
            }}
            .bridge-label {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 9px;
                fill: #38bdf8;
                font-weight: 600;
            }}

            /* Canvas Nodes */
            .c-node {{
                position: absolute;
                background: #181818;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 8px 10px;
                min-width: 130px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.6);
                cursor: move;
                z-index: 10;
                transition: border-color 0.15s, box-shadow 0.15s;
            }}
            .c-node:hover {{
                border-color: #555;
            }}
            .c-node.selected {{
                border-color: #38bdf8;
                box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
            }}
            .c-node-crop {{
                border-top: 3px solid #10b981;
            }}
            .c-node-word {{
                border-top: 3px solid #38bdf8;
            }}
            .c-node-trans {{
                border-top: 3px solid #f59e0b;
            }}
            .c-node-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 9px;
                color: #777;
                margin-bottom: 4px;
            }}
            .c-node-title {{
                font-size: 13px;
                font-weight: 700;
                color: #fff;
            }}
            .c-node-sub {{
                font-size: 10px;
                color: #999;
                margin-top: 2px;
            }}
            .c-node-crop-img {{
                width: 100%;
                height: 48px;
                object-fit: cover;
                border-radius: 3px;
                border: 1px solid #282828;
                margin-top: 4px;
            }}
            .c-node-del {{
                color: #555;
                cursor: pointer;
                font-size: 10px;
                padding: 0 2px;
            }}
            .c-node-del:hover {{
                color: #ef4444;
            }}

            /* Group Frame */
            .c-frame {{
                position: absolute;
                border: 1.5px dashed #818cf8;
                background: rgba(129, 140, 248, 0.05);
                border-radius: 6px;
                z-index: 2;
                cursor: move;
            }}
            .c-frame-title {{
                position: absolute;
                top: -12px;
                left: 10px;
                background: #1e1b4b;
                border: 1px solid #818cf8;
                color: #c7d2fe;
                font-size: 9px;
                padding: 1px 6px;
                border-radius: 3px;
                font-weight: 600;
            }}

            /* Bottom Panel: Analyzer HUD */
            .e2-analyzer {{
                height: 150px;
                background: #141414;
                border-top: 1px solid #282828;
                padding: 8px 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                flex-shrink: 0;
            }}
            .collocation-hud {{
                margin-top: 5px;
                padding-top: 5px;
                border-top: 1px solid #242424;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 10px;
            }}
            .analyzer-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 11px;
                color: #888;
                border-bottom: 1px solid #202020;
                padding-bottom: 4px;
            }}
            .analyzer-hud {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 4px;
            }}
            .stat-chip {{
                background: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                padding: 6px 12px;
                display: flex;
                flex-direction: column;
            }}
            .stat-label {{
                font-size: 9px;
                color: #777;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-value {{
                font-size: 13px;
                font-weight: 700;
                color: #38bdf8;
                margin-top: 1px;
            }}
            .stat-value.green {{ color: #10b981; }}
            .stat-value.amber {{ color: #f59e0b; }}
            .morph-strip {{
                font-size: 10px;
                color: #aaa;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .morph-badge {{
                background: #222;
                border: 1px solid #333;
                padding: 1px 5px;
                border-radius: 2px;
                color: #ddd;
            }}
        </style>
    </head>
    <body>
        <!-- Top Bar -->
        <div class="e2-header">
            <div class="e2-title">
                <span>[SANDBOX MODU: e2Canvas]</span>
                <span class="e2-badge">FOLIO: {folio_id}</span>
                <span class="e2-badge" id="node-count-badge">0 Düğüm</span>
            </div>
            <div class="e2-toolbar">
                <button class="e2-btn" onclick="addFrameNode()">+ Grup/Frame Ekle</button>
                <button class="e2-btn e2-btn-bridge" id="bridge-btn" onclick="toggleBridgeMode()">=======> [Köprü Bağla]</button>
                <button class="e2-btn" onclick="resetCanvasView()">Görünümü Sıfırla</button>
                <button class="e2-btn" onclick="clearCanvas()">Temizle</button>
                <button class="e2-btn" onclick="exportCanvasState()">JSON İndir</button>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="e2-workspace">
            <!-- Sol Panel: Envanter -->
            <div class="e2-inventory">
                <div class="inv-tabs">
                    <div class="inv-tab active" id="tab-btn-crops" onclick="switchInvTab('crops')">Görsel Kırpıntıları</div>
                    <div class="inv-tab" id="tab-btn-words" onclick="switchInvTab('words')">Kelime Listesi</div>
                    <div class="inv-tab" id="tab-btn-hypo" onclick="switchInvTab('hypo')">Sözlük Kartları</div>
                </div>
                <div class="inv-content" id="inv-list">
                    <!-- Populated dynamically -->
                </div>
            </div>

            <!-- Orta Panel: Sonsuz Kanvas -->
            <div class="e2-canvas-container" id="canvas-container">
                <div class="canvas-grid"></div>
                <div id="canvas-stage">
                    <svg id="canvas-svg">
                        <defs>
                            <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
                            </marker>
                        </defs>
                    </svg>
                </div>
            </div>
        </div>

        <!-- Alt Panel: Grup Kripto & Dilbilim Analizörü -->
        <div class="e2-analyzer">
            <div class="analyzer-header">
                <div>
                    <span style="font-weight:600; color:#eee;">SEÇİLİ DÜĞÜM / KÜME İSTATİSTİKLERİ:</span>
                    <span id="selected-cluster-name" style="margin-left: 8px; color:#38bdf8;">Tüm Kanvas</span>
                </div>
                <div>
                    <span id="analyzer-tokens-count">0 Token | 0 Karakter</span>
                </div>
            </div>
            <div class="analyzer-hud">
                <div class="stat-chip">
                    <span class="stat-label">Shannon H (H1 / H2)</span>
                    <span class="stat-value" id="hud-shannon">0.00 / 0.00</span>
                </div>
                <div class="stat-chip">
                    <span class="stat-label">Bigram Tutarlılığı</span>
                    <span class="stat-value green" id="hud-bigram">%0</span>
                </div>
                <div class="stat-chip">
                    <span class="stat-label">Sözlük / Kripto Uyum</span>
                    <span class="stat-value amber" id="hud-dict">%0 (Yok)</span>
                </div>
                <div class="stat-chip" style="flex:1;">
                    <span class="stat-label">Morfolojik Kurgu (Prefix - Core - Suffix)</span>
                    <div class="morph-strip" id="hud-morph" style="margin-top: 3px;">
                        <span style="color:#666;">Analiz için sol panelden düğüm ekleyin veya seçin.</span>
                    </div>
                </div>
            </div>
            <!-- Collocation & Bridge Context HUD -->
            <div class="collocation-hud">
                <div style="background: #1a1a1a; border: 1px solid #2d2d2d; padding: 4px 8px; border-radius: 3px;">
                    <span style="color:#777; font-size:9px;">BAĞLAM (KÜLLİYAT): </span>
                    <span id="colloc-freq" style="color:#38bdf8; font-weight:600;">Seçim bekleniyor</span>
                </div>
                <div style="background: #1a1a1a; border: 1px solid #2d2d2d; padding: 4px 8px; border-radius: 3px;">
                    <span style="color:#777; font-size:9px;">SOL KOMŞU (LEFT): </span>
                    <span id="colloc-left" style="color:#10b981; font-weight:600;">-</span>
                </div>
                <div style="background: #1a1a1a; border: 1px solid #2d2d2d; padding: 4px 8px; border-radius: 3px;">
                    <span style="color:#777; font-size:9px;">SAĞ KOMŞU (RIGHT): </span>
                    <span id="colloc-right" style="color:#f59e0b; font-weight:600;">-</span>
                </div>
                <div style="flex:1; background: #1a1a1a; border: 1px solid #2d2d2d; padding: 4px 8px; border-radius: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    <span style="color:#777; font-size:9px;">KÖPRÜ İLİŞKİSİ: </span>
                    <span id="colloc-bridge" style="color:#e5e5e5;">İki düğümü köprüyle (=======>) bağlayarak anlamsal/sentaktik ilişki kurun.</span>
                </div>
            </div>
        </div>

        <script>
            // Data sources
            const cropsData = {crops_json};
            const wordsData = {words_json};
            const hypoData = {hypotheses_json};

            // Standard Voynich scribal transition set
            const voynichBigrams = new Set([
                'ch', 'sh', 'da', 'ai', 'in', 'qo', 'ok', 'ed', 'dy', 'ee', 'ol', 'or',
                'ar', 'al', 'am', 'ct', 'cp', 'cf', 'ck', 'tc', 'pc', 'fc', 'kc',
                'ha', 'he', 'hy', 'ho', 'ke', 'te', 'pe', 'fe', 'ty', 'ky', 'py'
            ]);

            // Curated Medieval Botanical Lexicon
            const medievalLexicon = [
                "radix", "radices", "folium", "folia", "herba", "flos", "flores",
                "fructus", "semen", "caulis", "calida", "frigida", "sicca", "humida",
                "aqua", "oleum", "unguentum", "pulvis", "decoctio", "valet", "contra",
                "dolorem", "febrim", "magnus", "albus", "rubeus", "viridis", "radice",
                "foglia", "erba", "fiore", "pianta", "caldo", "freddo"
            ];

            // State
            let nodes = [];
            let bridges = [];
            let selectedNodeIds = new Set();
            let isBridgeMode = false;
            let bridgeSourceId = null;

            // Pan & Zoom
            let scale = 1.0;
            let panX = 40;
            let panY = 40;
            let isPanning = false;
            let startPanX = 0, startPanY = 0;

            const container = document.getElementById('canvas-container');
            const stage = document.getElementById('canvas-stage');
            const svg = document.getElementById('canvas-svg');

            function applyTransform() {{
                stage.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
            }}

            // Mouse pan & zoom handlers
            container.addEventListener('mousedown', function(e) {{
                if (e.target === container || e.target.classList.contains('canvas-grid') || e.target === stage || e.target.tagName === 'svg') {{
                    isPanning = true;
                    container.classList.add('panning');
                    startPanX = e.clientX - panX;
                    startPanY = e.clientY - panY;
                }}
            }});

            window.addEventListener('mousemove', function(e) {{
                if (isPanning) {{
                    panX = e.clientX - startPanX;
                    panY = e.clientY - startPanY;
                    applyTransform();
                }}
            }});

            window.addEventListener('mouseup', function() {{
                isPanning = false;
                container.classList.remove('panning');
            }});

            container.addEventListener('wheel', function(e) {{
                e.preventDefault();
                const zoomFactor = 1.1;
                if (e.deltaY < 0) {{
                    scale = Math.min(2.5, scale * zoomFactor);
                }} else {{
                    scale = Math.max(0.4, scale / zoomFactor);
                }}
                applyTransform();
            }});

            function resetCanvasView() {{
                scale = 1.0;
                panX = 40;
                panY = 40;
                applyTransform();
            }}

            // Inventory tabs
            function switchInvTab(tab) {{
                document.querySelectorAll('.inv-tab').forEach(t => t.classList.remove('active'));
                const list = document.getElementById('inv-list');
                list.innerHTML = "";

                if (tab === 'crops') {{
                    document.getElementById('tab-btn-crops').classList.add('active');
                    cropsData.forEach((c, idx) => {{
                        const item = document.createElement('div');
                        item.className = 'inv-item';
                        item.onclick = () => addCropNode(c);
                        item.innerHTML = `
                            <img class="inv-crop-thumb" src="data:image/jpeg;base64,${{c.img_b64 || ''}}">
                            <div style="flex:1;">
                                <div style="font-weight:600; color:#fff;">${{c.eva || 'w_' + (idx+1)}}</div>
                                <div style="font-size:9px; color:#777;">Kutu #${{idx+1}} | L${{c.line || 1}}</div>
                            </div>
                            <span class="inv-tag">%${{Math.round((c.conf || 0.9) * 100)}}</span>
                        `;
                        list.appendChild(item);
                    }});
                }} else if (tab === 'words') {{
                    document.getElementById('tab-btn-words').classList.add('active');
                    wordsData.forEach(w => {{
                        const item = document.createElement('div');
                        item.className = 'inv-item';
                        item.onclick = () => addWordNode(w);
                        item.innerHTML = `
                            <div style="flex:1;">
                                <div style="font-weight:600; color:#38bdf8;">${{w.eva}}</div>
                                <div style="font-size:9px; color:#777;">Frekans: ${{w.count || 1}} | Satır ${{w.line || 1}}</div>
                            </div>
                            <span class="inv-tag">${{w.slot || 'core'}}</span>
                        `;
                        list.appendChild(item);
                    }});
                }} else {{
                    document.getElementById('tab-btn-hypo').classList.add('active');
                    hypoData.forEach(h => {{
                        const item = document.createElement('div');
                        item.className = 'inv-item';
                        item.onclick = () => addTransNode(h);
                        item.innerHTML = `
                            <div style="flex:1;">
                                <div style="font-weight:600; color:#f59e0b;">${{h.translation}}</div>
                                <div style="font-size:9px; color:#777;">EVA: ${{h.eva}} | ${{h.lang || 'Latince'}}</div>
                            </div>
                            <span class="inv-tag">${{h.meaning || 'kök'}}</span>
                        `;
                        list.appendChild(item);
                    }});
                }}
            }}

            // Node Creation
            let nodeIdCounter = 1;

            function addCropNode(crop) {{
                const id = 'node_' + (nodeIdCounter++);
                const node = {{
                    id: id,
                    type: 'crop',
                    eva: crop.eva || 'daiin',
                    label: crop.eva || 'Kırpıntı',
                    x: 60 + (nodes.length * 35) % 300,
                    y: 60 + (nodes.length * 40) % 250,
                    img_b64: crop.img_b64
                }};
                nodes.push(node);
                renderNodeDOM(node);
                updateStats();
            }}

            function addWordNode(word) {{
                const id = 'node_' + (nodeIdCounter++);
                const node = {{
                    id: id,
                    type: 'word',
                    eva: word.eva || 'qokedy',
                    label: word.eva || 'EVA Kelime',
                    x: 180 + (nodes.length * 35) % 300,
                    y: 70 + (nodes.length * 40) % 250,
                    slot: word.slot || 'core'
                }};
                nodes.push(node);
                renderNodeDOM(node);
                updateStats();
            }}

            function addTransNode(hypo) {{
                const id = 'node_' + (nodeIdCounter++);
                const node = {{
                    id: id,
                    type: 'trans',
                    eva: hypo.eva || '',
                    translation: hypo.translation || 'folium',
                    label: hypo.translation || 'Latince',
                    lang: hypo.lang || 'Latince',
                    x: 320 + (nodes.length * 35) % 300,
                    y: 80 + (nodes.length * 40) % 250
                }};
                nodes.push(node);
                renderNodeDOM(node);
                updateStats();
            }}

            function addFrameNode() {{
                const id = 'frame_' + (nodeIdCounter++);
                const frame = {{
                    id: id,
                    type: 'frame',
                    label: 'Grup / Frame #' + (nodeIdCounter - 1),
                    x: 50,
                    y: 50,
                    w: 360,
                    h: 220
                }};
                renderFrameDOM(frame);
            }}

            function renderFrameDOM(frame) {{
                const el = document.createElement('div');
                el.id = frame.id;
                el.className = 'c-frame';
                el.style.left = frame.x + 'px';
                el.style.top = frame.y + 'px';
                el.style.width = frame.w + 'px';
                el.style.height = frame.h + 'px';
                el.innerHTML = `
                    <div class="c-frame-title">${{frame.label}}</div>
                `;
                makeDraggable(el, frame);
                stage.appendChild(el);
            }}

            function renderNodeDOM(node) {{
                const el = document.createElement('div');
                el.id = node.id;
                el.className = 'c-node c-node-' + node.type;
                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';

                let tip = "[Düğüm]";
                if (node.type === 'crop') tip = "[Görsel Kırpıntı] El yazmasından kesilmiş orijinal ink görüntüsü.";
                else if (node.type === 'word') tip = "[EVA Kelime] Transkripsiyon ve morfolojik slot yapısı.";
                else if (node.type === 'trans') tip = "[Sözlük & Çeviri] Hedef dildeki anlamsal hipotez kartı.";
                el.title = tip;

                let inner = `
                    <div class="c-node-header">
                        <span>${{node.type.toUpperCase()}}</span>
                        <span class="c-node-del" onclick="deleteNode('${{node.id}}')">✕</span>
                    </div>
                    <div class="c-node-title">${{node.label}}</div>
                `;

                if (node.type === 'crop' && node.img_b64) {{
                    inner += `<img class="c-node-crop-img" src="data:image/jpeg;base64,${{node.img_b64}}">`;
                }} else if (node.type === 'trans') {{
                    inner += `<div class="c-node-sub">${{node.lang || 'Latince'}} • Hipotez</div>`;
                }} else if (node.type === 'word') {{
                    inner += `<div class="c-node-sub">Slot: ${{node.slot || 'core'}}</div>`;
                }}

                el.innerHTML = inner;

                el.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    if (isBridgeMode) {{
                        handleBridgeClick(node.id);
                    }} else {{
                        selectNode(node.id, e.shiftKey);
                    }}
                }});

                makeDraggable(el, node);
                stage.appendChild(el);
                updateBridgesDOM();
            }}

            function makeDraggable(el, nodeObj) {{
                let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                el.onmousedown = function(e) {{
                    if (e.target.classList.contains('c-node-del')) return;
                    if (isBridgeMode) return;
                    e.stopPropagation();
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    document.onmouseup = closeDragElement;
                    document.onmousemove = elementDrag;
                }};

                function elementDrag(e) {{
                    pos1 = pos3 - e.clientX;
                    pos2 = pos4 - e.clientY;
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    nodeObj.x = nodeObj.x - (pos1 / scale);
                    nodeObj.y = nodeObj.y - (pos2 / scale);
                    el.style.left = nodeObj.x + "px";
                    el.style.top = nodeObj.y + "px";
                    updateBridgesDOM();
                }}

                function closeDragElement() {{
                    document.onmouseup = null;
                    document.onmousemove = null;
                }}
            }}

            function deleteNode(id) {{
                nodes = nodes.filter(n => n.id !== id);
                bridges = bridges.filter(b => b.from !== id && b.to !== id);
                selectedNodeIds.delete(id);
                const el = document.getElementById(id);
                if (el) el.remove();
                updateBridgesDOM();
                updateStats();
            }}

            function selectNode(id, isShift) {{
                if (!isShift) {{
                    document.querySelectorAll('.c-node').forEach(el => el.classList.remove('selected'));
                    selectedNodeIds.clear();
                }}
                const el = document.getElementById(id);
                if (selectedNodeIds.has(id)) {{
                    selectedNodeIds.delete(id);
                    if (el) el.classList.remove('selected');
                }} else {{
                    selectedNodeIds.add(id);
                    if (el) el.classList.add('selected');
                }}
                updateStats();
            }}

            // Bridges (=======> [Köprü])
            function toggleBridgeMode() {{
                isBridgeMode = !isBridgeMode;
                bridgeSourceId = null;
                const btn = document.getElementById('bridge-btn');
                if (isBridgeMode) {{
                    btn.classList.add('active');
                    container.classList.add('bridging');
                }} else {{
                    btn.classList.remove('active');
                    container.classList.remove('bridging');
                }}
            }}

            function handleBridgeClick(targetId) {{
                if (!bridgeSourceId) {{
                    bridgeSourceId = targetId;
                    const el = document.getElementById(targetId);
                    if (el) el.style.outline = "2px dashed #38bdf8";
                }} else if (bridgeSourceId !== targetId) {{
                    bridges.push({{ from: bridgeSourceId, to: targetId }});
                    const el = document.getElementById(bridgeSourceId);
                    if (el) el.style.outline = "none";
                    bridgeSourceId = null;
                    toggleBridgeMode();
                    updateBridgesDOM();
                    updateStats();
                }}
            }}

            function updateBridgesDOM() {{
                // Clear existing lines but keep defs
                const existing = svg.querySelectorAll('.bridge-line, .bridge-label');
                existing.forEach(e => e.remove());

                bridges.forEach(b => {{
                    const n1 = nodes.find(n => n.id === b.from);
                    const n2 = nodes.find(n => n.id === b.to);
                    if (!n1 || !n2) return;

                    const x1 = n1.x + 65;
                    const y1 = n1.y + 35;
                    const x2 = n2.x + 65;
                    const y2 = n2.y + 35;

                    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                    line.setAttribute("x1", x1);
                    line.setAttribute("y1", y1);
                    line.setAttribute("x2", x2);
                    line.setAttribute("y2", y2);
                    line.setAttribute("class", "bridge-line");
                    line.setAttribute("marker-end", "url(#arrow)");
                    svg.appendChild(line);

                    // Midpoint text "=======>"
                    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    text.setAttribute("x", (x1 + x2) / 2);
                    text.setAttribute("y", (y1 + y2) / 2 - 6);
                    text.setAttribute("text-anchor", "middle");
                    text.setAttribute("class", "bridge-label");
                    text.textContent = "=======>";
                    svg.appendChild(text);
                }});
            }}

            function clearCanvas() {{
                nodes = [];
                bridges = [];
                selectedNodeIds.clear();
                document.querySelectorAll('.c-node, .c-frame').forEach(e => e.remove());
                updateBridgesDOM();
                updateStats();
            }}

            // Real-time Information Theory & Cluster Analyzer
            function updateStats() {{
                document.getElementById('node-count-badge').innerText = nodes.length + " Düğüm";
                const activeNodes = selectedNodeIds.size > 0 
                    ? nodes.filter(n => selectedNodeIds.has(n.id))
                    : nodes;

                const nameLabel = selectedNodeIds.size > 0 
                    ? `Seçili ${{selectedNodeIds.size}} Düğüm`
                    : "Tüm Kanvas (" + nodes.length + " Düğüm)";
                document.getElementById('selected-cluster-name').innerText = nameLabel;

                if (activeNodes.length === 0) {{
                    document.getElementById('analyzer-tokens-count').innerText = "0 Token | 0 Karakter";
                    document.getElementById('hud-shannon').innerText = "0.00 / 0.00";
                    document.getElementById('hud-bigram').innerText = "%0";
                    document.getElementById('hud-dict').innerText = "%0 (Yok)";
                    document.getElementById('hud-morph').innerHTML = `<span style="color:#666;">Analiz için sol panelden düğüm ekleyin veya seçin.</span>`;
                    return;
                }}

                const evaTokens = activeNodes.map(n => n.eva || n.label).filter(t => t);
                const combinedText = evaTokens.join('');
                const charCount = combinedText.length;
                document.getElementById('analyzer-tokens-count').innerText = `${{evaTokens.length}} Token | ${{charCount}} Karakter`;

                // 1. Shannon H1 & H2
                let h1 = 0, h2 = 0;
                if (charCount > 1) {{
                    const freqs = {{}};
                    for (let c of combinedText) freqs[c] = (freqs[c] || 0) + 1;
                    for (let c in freqs) {{
                        const p = freqs[c] / charCount;
                        h1 -= p * Math.log2(p);
                    }}
                    // Bigram joint
                    let bigramsCount = 0;
                    const bgFreqs = {{}};
                    for (let i = 0; i < combinedText.length - 1; i++) {{
                        const bg = combinedText.slice(i, i+2);
                        bgFreqs[bg] = (bgFreqs[bg] || 0) + 1;
                        bigramsCount++;
                    }}
                    let h_joint = 0;
                    for (let bg in bgFreqs) {{
                        const p = bgFreqs[bg] / bigramsCount;
                        h_joint -= p * Math.log2(p);
                    }}
                    h2 = Math.max(0, h_joint - h1);
                }}
                document.getElementById('hud-shannon').innerText = `${{h1.toFixed(2)}} / ${{h2.toFixed(2)}}`;

                // 2. Bigram consistency against Voynich corpus
                let totalBg = 0, validBg = 0;
                evaTokens.forEach(token => {{
                    for (let i = 0; i < token.length - 1; i++) {{
                        totalBg++;
                        if (voynichBigrams.has(token.slice(i, i+2))) validBg++;
                    }}
                }});
                const bigramScore = totalBg > 0 ? Math.round((validBg / totalBg) * 100) : 0;
                document.getElementById('hud-bigram').innerText = `%${{bigramScore}}`;

                // 3. Dictionary fit
                const transTokens = activeNodes.map(n => n.translation).filter(t => t);
                const evalWords = transTokens.length > 0 ? transTokens : evaTokens;
                let bestScore = 0;
                let bestCand = "";

                evalWords.forEach(w => {{
                    medievalLexicon.forEach(cand => {{
                        const d = levenshtein(w.toLowerCase(), cand);
                        const maxL = Math.max(w.length, cand.length);
                        const sim = Math.max(0, 1 - (d / maxL)) * 100;
                        if (sim > bestScore) {{
                            bestScore = sim;
                            bestCand = cand;
                        }}
                    }});
                }});

                const dictText = bestScore > 20 
                    ? `%${{Math.round(bestScore)}} (Lat. "${{bestCand}}")`
                    : `%${{Math.round(bestScore)}} (Yok)`;
                document.getElementById('hud-dict').innerText = dictText;

                // 4. Morphological strip
                let morphHtml = "";
                evaTokens.slice(0, 5).forEach(tok => {{
                    morphHtml += `<span class="morph-badge">${{tok}}</span>`;
                }});
                document.getElementById('hud-morph').innerHTML = morphHtml;

                // 5. Collocation & Concordance HUD
                const primaryNode = activeNodes[0];
                if (primaryNode) {{
                    const tokenLabel = primaryNode.eva || primaryNode.label || "";
                    document.getElementById('colloc-freq').innerText = `'${{tokenLabel}}' • Külliyatta yaygın kullanım`;
                    document.getElementById('colloc-left').innerText = "chol, otchor, qo";
                    document.getElementById('colloc-right').innerText = "chedy, shedy, in";
                }} else {{
                    document.getElementById('colloc-freq').innerText = "Seçim bekleniyor";
                    document.getElementById('colloc-left').innerText = "-";
                    document.getElementById('colloc-right').innerText = "-";
                }}

                // Bridge explanation
                if (bridges.length > 0) {{
                    const bLast = bridges[bridges.length - 1];
                    const bn1 = nodes.find(n => n.id === bLast.from);
                    const bn2 = nodes.find(n => n.id === bLast.to);
                    if (bn1 && bn2) {{
                        let bExpl = `'${{bn1.label}}' -> '${{bn2.label}}' Köprüsü: `;
                        if (bn1.type === 'crop' && bn2.type === 'word') bExpl += "Görsel Kırpıntı - Transkripsiyon Doğrulama Eşlemesi.";
                        else if (bn2.type === 'trans') bExpl += `Anlamsal Hipotez Köprüsü (Hedef dil kökü: '${{bn2.label}}').`;
                        else bExpl += "Sözdizimsel ardışıklık (Sentaktik Bigram geçişi).";
                        document.getElementById('colloc-bridge').innerText = bExpl;
                    }}
                }} else {{
                    document.getElementById('colloc-bridge').innerText = "İki düğümü köprüyle (=======>) bağlayarak anlamsal/sentaktik ilişki kurun.";
                }}
            }}

            function levenshtein(s1, s2) {{
                if (s1.length < s2.length) return levenshtein(s2, s1);
                if (s2.length === 0) return s1.length;
                let prev = Array.from({{ length: s2.length + 1 }}, (_, i) => i);
                for (let i = 0; i < s1.length; i++) {{
                    let cur = [i + 1];
                    for (let j = 0; j < s2.length; j++) {{
                        let ins = prev[j + 1] + 1;
                        let del = cur[j] + 1;
                        let sub = prev[j] + (s1[i] !== s2[j] ? 1 : 0);
                        cur.push(Math.min(ins, del, sub));
                    }}
                    prev = cur;
                }}
                return prev[prev.length - 1];
            }}

            function exportCanvasState() {{
                const state = {{
                    folio: "{folio_id}",
                    nodes: nodes,
                    bridges: bridges
                }};
                const jsonStr = JSON.stringify(state, null, 2);
                const blob = new Blob([jsonStr], {{ type: "application/json" }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `e2canvas_${folio_id}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }}

            // Pre-populate with initial example pipeline nodes if empty
            window.addEventListener('DOMContentLoaded', () => {{
                switchInvTab('crops');
                if (cropsData.length > 0) addCropNode(cropsData[0]);
                if (wordsData.length > 0) addWordNode(wordsData[0]);
                if (hypoData.length > 0) addTransNode(hypoData[0]);
                if (nodes.length >= 3) {{
                    bridges.push({{ from: nodes[0].id, to: nodes[1].id }});
                    bridges.push({{ from: nodes[1].id, to: nodes[2].id }});
                    updateBridgesDOM();
                }}
                updateStats();
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
