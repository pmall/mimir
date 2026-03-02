"""
Generate an interactive HTML visualization of a target's fingerprint mask.

Usage:
    uv run python -m scripts.dataset.visualize_fingerprint \
        -i data/run78-v2/features_targets \
        -t A0A024RBG1 \
        -o data/visualizations
"""

import argparse
import gzip
import json
import logging
import re
import sys
import tarfile
from pathlib import Path

import lmdb
import msgpack

from mimir.features import (
    TargetFeatures, 
    compute_rsasa, 
    get_smoothed_rsasa, 
    get_fingerprint_mask
)

logger = logging.getLogger(__name__)


# ---
# Embedded HTML Template
# ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Target Fingerprint: {{ TARGET_ID }}</title>
    <!-- Use 3Dmol.js from CDN -->
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        header { padding: 15px 20px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; flex: 0 0 auto; }
        h1 { font-size: 20px; margin: 0; color: #e2e8f0; }
        p.subtitle { color: #94a3b8; margin: 0; font-size: 14px; }
        
        /* 1D Scroll View (Top Half) */
        .chart-wrapper { width: 100%; height: 35vh; overflow-x: auto; overflow-y: hidden; background: #0f172a; border-bottom: 2px solid #334155; flex: 0 0 auto; display: flex; align-items: center; position: relative; }
        .chart-container { display: flex; position: relative; width: max-content; padding: 0 20px; height: 100%; align-items: center; }
        
        .col { display: flex; flex-direction: column; width: 28px; align-items: center; position: relative; z-index: 2; margin: 0 1px; cursor: pointer; }
        .hover-target { position: absolute; inset: 0; z-index: 10; cursor: pointer; border-radius: 4px; transition: background 0.1s; }
        .hover-target:hover, .col.active .hover-target { background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.3); box-sizing: border-box; }
        
        .rsasa-cell { height: 10vh; width: 100%; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 8px; gap: 2px; }
        .rsasa-bar { width: 10px; border-radius: 3px 3px 0 0; }
        .smoothed-rsasa-bar { width: 10px; border-radius: 3px 3px 0 0; }
        
        .seq-cell { height: 26px; width: 24px; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: bold; border-radius: 4px; }
        
        .plddt-cell { height: 10vh; width: 100%; display: flex; align-items: flex-start; justify-content: center; padding-top: 8px; }
        .plddt-bar { width: 14px; border-radius: 0 0 3px 3px; }
        
        /* Threshold Lines - we anchor relative to the full wrapper */
        .lines-container { position: absolute; top: 0; bottom: 0; left: 0; right: 0; z-index: 1; pointer-events: none; }
        .thresh-rsasa { position: absolute; height: 1px; width: 100%; border-top: 1px dashed #94a3b8; opacity: 0.5; left: 0; } 
        .thresh-plddt { position: absolute; height: 1px; width: 100%; border-top: 1px solid #ef4444; opacity: 0.5; left: 0; top: calc(50% + 14px + 7vh); } /* top down, +14px for half sequence box + 7vh representing (100-70)=30 down but plddt goes 0..100 mapping to 10vh height, so 70 is 7vh down from the center start point */
        
        /* Masked IN */
        .col.in .rsasa-bar { background: #0ea5e9; }
        .col.in .smoothed-rsasa-bar { background: #8b5cf6; } /* Purple for smoothed */
        .col.in .seq-cell { background: #3b82f6; color: #ffffff; box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }
        .col.in .plddt-bar { background: #f59e0b; }
        
        /* Masked OUT */
        .col.out .rsasa-bar { background: #475569; opacity: 0.3; }
        .col.out .smoothed-rsasa-bar { background: #475569; opacity: 0.3; }
        .col.out .seq-cell { background: #334155; color: #64748b; }
        .col.out .plddt-bar { background: #475569; opacity: 0.3; }
        
        /* 3D Viewer (Bottom Half) */
        #viewer-container { flex: 1 1 auto; position: relative; background: #000; width: 100%; }
        #viewer3d { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
        
        /* Tooltip */
        .tooltip { position: absolute; top: 20px; right: 20px; width: 220px; background: rgba(15, 23, 42, 0.9); border: 1px solid #475569; border-radius: 8px; padding: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: none; backdrop-filter: blur(4px); z-index: 100; pointer-events: none; }
        .tooltip h2 { margin: 0 0 8px 0; font-size: 15px; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 6px; display: flex; justify-content: space-between; }
        .tooltip .row { display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }
        .tooltip .label { color: #94a3b8; }
        .tooltip .value { font-weight: 600; color: #f8fafc; font-family: monospace; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge.in { background: rgba(22, 101, 52, 0.8); color: #4ade80; }
        .badge.out { background: rgba(127, 29, 29, 0.8); color: #f87171; }
        
        /* Floating HUD inside 3D viewer */
        .hud { position: absolute; top: 20px; left: 20px; z-index: 10; pointer-events: none; }
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Target Fingerprint: {{ TARGET_ID }}</h1>
            <p class="subtitle">1D sequence track mapping to 3D structure.</p>
        </div>
        <div style="display: flex; gap: 24px; align-items: center; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 8px; border: 1px solid #334155;">
            <div>
                <p class="subtitle" style="font-size: 11px; margin-bottom: 4px;"><strong>LEGEND</strong></p>
                <div style="display: flex; gap: 12px; font-size: 11px; color: #cbd5e1;">
                    <span style="display: flex; align-items: center; gap: 4px;"><span style="width: 8px; height: 8px; background: #0ea5e9; display: inline-block;"></span> rSASA</span>
                    <span style="display: flex; align-items: center; gap: 4px;"><span style="width: 8px; height: 8px; background: #8b5cf6; display: inline-block;"></span> Smoothed rSASA</span>
                </div>
            </div>
            <div style="height: 24px; width: 1px; background: #475569;"></div>
            <div>
                <p class="subtitle" style="font-size: 11px; margin-bottom: 4px;"><strong>THRESHOLDS</strong></p>
                <div style="display: flex; gap: 12px; font-size: 11px; color: #cbd5e1;">
                    <span style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 1px; border-top: 1px solid #ef4444; display: inline-block;"></span> pLDDT = 70.0</span>
                    <span style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 1px; border-top: 1px dashed #94a3b8; display: inline-block;"></span> Smoothed rSASA = <span id="legend-rsasa-thresh">N/A</span></span>
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <p class="subtitle" style="color: #cbd5e1; font-size: 13px;">Hover/Click boxes to highlight 3D</p>
            <p class="subtitle" style="font-size: 11px; color: #4ade80;">Green atoms = Kept Fingerprint</p>
        </div>
    </header>

    <div class="chart-wrapper" id="scroll-wrapper">
        <div class="chart-container" id="chart">
            <div class="lines-container">
                <div class="thresh-rsasa"></div>
                <div class="thresh-plddt"></div>
            </div>
            <!-- Columns injected by JS -->
        </div>
    </div>

    <div id="viewer-container">
        <div id="viewer3d"></div>
        <div class="hud tooltip" id="tooltip">
            <h2><span>Pos <span id="tt-pos"></span></span> <span id="tt-aa"></span></h2>
            <div class="row"><span class="label">Status</span> <span id="tt-status" class="badge"></span></div>
            <div class="row"><span class="label">rSASA</span> <span id="tt-rsasa" class="value"></span></div>
            <div class="row"><span class="label">Sm. rSASA</span> <span id="tt-smoothed-rsasa" class="value" style="color: #c4b5fd;"></span></div>
            <div class="row"><span class="label">pLDDT</span> <span id="tt-plddt" class="value"></span></div>
        </div>
    </div>

    <script>
        const DATA = {{ JSON_DATA }};
        const RAW_CIF = `{{ CIF_DATA }}`;
        
        const chart = document.getElementById('chart');
        const tooltip = document.getElementById('tooltip');
        let viewer = null;
        let selectedCol = null;
        
        // --- 1D INITIALIZATION ---
        
        function init1D() {
            // Find max height pixel values derived from CSS (vh units change)
            const rsasaMaxHt = (window.innerHeight * 0.10); // 10vh
            const plddtMaxHt = (window.innerHeight * 0.10); // 10vh

            // Setup dynamic rsasa threshold line position
            if (DATA.rsasa_threshold !== null) {
                const rsasaLine = document.querySelector('.thresh-rsasa');
                // Calculate position relative to bottom just like the old static calculation 
                // but using the dynamic threshold
                const vhVal = DATA.rsasa_threshold * 10;
                rsasaLine.style.bottom = `calc(50% + 14px + ${vhVal}vh)`;
                document.getElementById('legend-rsasa-thresh').textContent = DATA.rsasa_threshold.toFixed(2);
            } else {
                // hide it if None
                document.querySelector('.thresh-rsasa').style.display = 'none';
                document.getElementById('legend-rsasa-thresh').textContent = "None (fits)";
            }

            for (let i = 0; i < DATA.sequence.length; i++) {
                const aa = DATA.sequence[i];
                const rsasa = DATA.rsasa[i];
                const smoothed_rsasa = DATA.smoothed_rsasa[i];
                const plddt = DATA.plddt[i];
                const isMaskedIn = DATA.mask[i];
                const pos = DATA.positions[i]; // 1-indexed AlphaFold / mmCIF sequence position
                
                const rsasaHeight = Math.min(rsasa, 1.0) * rsasaMaxHt;
                const smoothedRsasaHeight = Math.min(smoothed_rsasa, 1.0) * rsasaMaxHt;
                const plddtHeight = (plddt / 100) * plddtMaxHt;
                
                const col = document.createElement('div');
                col.className = 'col ' + (isMaskedIn ? 'in' : 'out');
                col.id = `col-${pos}`;
                
                col.innerHTML = `
                    <div class="hover-target"></div>
                    <div class="rsasa-cell">
                        <div class="rsasa-bar" style="height: ${rsasaHeight}px;"></div>
                        <div class="smoothed-rsasa-bar" style="height: ${smoothedRsasaHeight}px;"></div>
                    </div>
                    <div class="seq-cell">${aa}</div>
                    <div class="plddt-cell"><div class="plddt-bar" style="height: ${plddtHeight}px;"></div></div>
                `;
                
                // Interaction
                const hoverTarget = col.querySelector('.hover-target');
                hoverTarget.addEventListener('mouseenter', () => highlightPos(i));
                hoverTarget.addEventListener('mouseleave', () => unhighlightPos());
                
                chart.appendChild(col);
            }
        }
        
        // --- 3D INITIALIZATION ---
        
        function init3D() {
            let element = document.getElementById('viewer3d');
            let config = { backgroundColor: '#000000' };
            viewer = $3Dmol.createViewer( element, config );
            
            viewer.addModel( RAW_CIF, "cif" );
            
            // Base Style (Skipped): Grey spheres for everything
            viewer.setStyle({}, {
                sphere: { color: '#334155', opacity: 0.6 }
            });
            
            // Highlight Style (Kept Fingerprint): Opaque colored spheres
            const keptResidues = [];
            for (let i = 0; i < DATA.mask.length; i++) {
                if (DATA.mask[i]) {
                    keptResidues.push(DATA.positions[i]);
                }
            }
            
            if (keptResidues.length > 0) {
                viewer.setStyle({resi: keptResidues}, {
                    sphere: { color: '#22c55e', opacity: 1.0 }
                });
            }
            
            viewer.zoomTo();
            viewer.render();
            
            // Map 3D clicks back to 1D scroll tracking
            viewer.setClickable({}, true, function(atom, viewer, event, container) {
                if(atom.resi) {
                    const idx = DATA.positions.indexOf(parseInt(atom.resi));
                    if (idx !== -1) {
                        const colEl = document.getElementById(`col-${atom.resi}`);
                        if (colEl) {
                            colEl.scrollIntoView({behavior: "smooth", block: "center", inline: "center"});
                            highlightPos(idx);
                            setTimeout(unhighlightPos, 3000); // Auto-clear after click
                        }
                    }
                }
            });
        }
        
        // --- INTERACTION LOGIC ---
        
        let highlightStyle = null; // Store previous highlight state
        
        function highlightPos(idx) {
            const aa = DATA.sequence[idx];
            const rsasa = DATA.rsasa[idx];
            const smoothed_rsasa = DATA.smoothed_rsasa[idx];
            const plddt = DATA.plddt[idx];
            const isMaskedIn = DATA.mask[idx];
            const pos = DATA.positions[idx];
            
            // Update 1D Tooltip
            document.getElementById('tt-pos').textContent = pos;
            document.getElementById('tt-aa').textContent = aa;
            document.getElementById('tt-rsasa').textContent = rsasa.toFixed(3);
            document.getElementById('tt-smoothed-rsasa').textContent = smoothed_rsasa.toFixed(3);
            document.getElementById('tt-plddt').textContent = plddt.toFixed(1);
            
            const statusEl = document.getElementById('tt-status');
            statusEl.textContent = isMaskedIn ? 'KEPT' : 'SKIPPED';
            statusEl.className = 'badge ' + (isMaskedIn ? 'in' : 'out');
            
            tooltip.style.display = 'block';
            
            // Update 1D Active Column state
            if (selectedCol) selectedCol.classList.remove('active');
            selectedCol = document.getElementById(`col-${pos}`);
            if (selectedCol) selectedCol.classList.add('active');
            
            // Highlight 3D: Use addStyle to overlay a larger yellow sphere
            viewer.addStyle({resi: [pos]}, {sphere: {scale: 1.3, color: '#eab308', opacity: 1.0}});
            viewer.render();
        }
        
        function unhighlightPos() {
            tooltip.style.display = 'none';
            if (selectedCol) selectedCol.classList.remove('active');
            
            // Reset 3D: Reapply base styles setting to wipe out addStyle
            const keptResidues = [];
            for (let i = 0; i < DATA.mask.length; i++) {
                if (DATA.mask[i]) keptResidues.push(DATA.positions[i]);
            }
            
            viewer.setStyle({}, { sphere: { color: '#334155', opacity: 0.6 } });
            if (keptResidues.length > 0) {
                viewer.setStyle({resi: keptResidues}, {
                    sphere: { color: '#22c55e', opacity: 1.0 }
                });
            }
            viewer.render();
        }

        // Bootstrap
        window.onload = function() {
            init1D();
            init3D();
        };
    </script>
</body>
</html>
"""


# ---
# Main Script
# ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization of a target's fingerprint."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input features targets LMDB",
    )
    parser.add_argument(
        "--tar-file",
        type=Path,
        required=True,
        help="Path to the AlphaFold2 bulk tarball (e.g. data/UP000005640_9606_HUMAN_v6.tar)",
    )
    parser.add_argument(
        "-t", "--target-id",
        type=str,
        required=True,
        help="Target entry ID to visualize (e.g. A0A024RBG1)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to save the output HTML file",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=280,
        help="Maximum number of positions to keep in the fingerprint (default: 280)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not args.input.exists():
        logger.error(f"Input LMDB not found: {args.input}")
        sys.exit(1)
        
    if not args.tar_file.exists():
        logger.error(f"AlphaFold tarball not found: {args.tar_file}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.output

    logger.info(f"Opening LMDB to fetch {args.target_id}...")
    
    env = lmdb.open(str(args.input), readonly=True, lock=False)
    with env.begin() as txn:
        # Check if the target exists
        raw_val = txn.get(args.target_id.encode("utf-8"))
        if raw_val is None:
            logger.error(f"Target '{args.target_id}' not found in LMDB.")
            sys.exit(1)

        raw_dict = msgpack.unpackb(raw_val)
        target = TargetFeatures.from_dict(raw_dict)

    env.close()

    logger.info(f"Loaded target {args.target_id} (Length: {len(target.sequence)})")

    # 1. Calculate values
    rsasa_np = compute_rsasa(target.sequence, target.sasa)
    smoothed_rsasa_np = get_smoothed_rsasa(rsasa_np, window_size=15)
    
    # 2. Get Boolean Mask and Threshold
    mask_result = get_fingerprint_mask(
        sequence=target.sequence,
        sasa=target.sasa,
        plddt=target.residue_plddt,
        max_len=args.max_len
    )
    
    mask_np, threshold = mask_result
    
    if mask_np is None:
        logger.warning(
            f"Target '{args.target_id}' fingerprint is shorter than 15 valid positions. "
            f"Highlighting none."
        )
        mask_list = [False] * len(target.sequence)
    else:
        mask_list = mask_np.tolist()

    # 3. Extract RAW CIF from tarball
    logger.info(f"Extracting 3D .cif text from {args.tar_file}...")
    cif_text = ""
    # regex matches: AF-A0A024R1R8-F1-model_v6.cif.gz
    pattern = re.compile(rf"AF-{args.target_id}-F1-model_v\d+\.cif\.gz")
    
    with tarfile.open(args.tar_file, "r|") as tar:
        for item in tar:
            if pattern.search(item.name):
                f = tar.extractfile(item)
                if f:
                    cif_bytes = gzip.decompress(f.read())
                    cif_text = cif_bytes.decode("utf-8")
                break
                
    if not cif_text:
        logger.error(f"Could not find matching .cif.gz file for {args.target_id} in tarball.")
        sys.exit(1)

    # 4. Create JSON payload
    data_payload = {
        "sequence": target.sequence,
        "positions": target.position_ids,
        "rsasa": rsasa_np.tolist(),
        "smoothed_rsasa": smoothed_rsasa_np.tolist(),
        "plddt": target.residue_plddt,
        "mask": mask_list,
        "rsasa_threshold": threshold,
    }

    logger.info("Injecting data into HTML template...")
    html_content = HTML_TEMPLATE.replace("{{ TARGET_ID }}", args.target_id)
    html_content = html_content.replace("{{ JSON_DATA }}", json.dumps(data_payload))
    
    # Escape backticks and problematic chars in CIF text just in case before JS injection
    safe_cif = cif_text.replace("`", "'").replace("\\", "\\\\")
    html_content = html_content.replace('{{ CIF_DATA }}', safe_cif)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Successfully generated visualization: {out_file}")


if __name__ == "__main__":
    main()
