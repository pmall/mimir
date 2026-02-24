"""
Generate an interactive HTML visualization of a target's fingerprint mask.

Usage:
    uv run python -m scripts.dataset.visualize_fingerprint \
        -i data/run78-v2/features_targets \
        -t A0A024RBG1 \
        -o data/visualizations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import lmdb
import msgpack

from mimir.structure_features import TargetFeatures, compute_rsasa, get_fingerprint_mask

logger = logging.getLogger(__name__)


# ---
# Embedded HTML Template
# ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fingerprint Visualization: {{ TARGET_ID }}</title>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; overflow-y: hidden; }
        h1 { font-size: 24px; margin-bottom: 5px; color: #e2e8f0; }
        p.subtitle { color: #94a3b8; margin-top: 0; margin-bottom: 30px; }
        
        .chart-wrapper { width: 100%; overflow-x: auto; background: #1e293b; border-radius: 12px; padding: 20px 0; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
        .chart-container { display: flex; position: relative; width: max-content; padding: 0 20px; }
        
        .col { display: flex; flex-direction: column; width: 22px; align-items: center; position: relative; z-index: 2; margin: 0 1px; }
        
        /* The invisible hover overlay for each column */
        .hover-target { position: absolute; inset: 0; z-index: 10; cursor: crosshair; }
        .hover-target:hover { background: rgba(255, 255, 255, 0.05); }
        
        .rsasa-cell { height: 120px; width: 100%; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 8px; }
        .rsasa-bar { width: 14px; border-radius: 3px 3px 0 0; }
        
        .seq-cell { height: 26px; width: 22px; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: bold; border-radius: 4px; }
        
        .plddt-cell { height: 120px; width: 100%; display: flex; align-items: flex-start; justify-content: center; padding-top: 8px; }
        .plddt-bar { width: 14px; border-radius: 0 0 3px 3px; }
        
        /* Threshold Lines */
        .lines-container { position: absolute; top: 0; bottom: 0; left: 20px; right: 20px; z-index: 1; pointer-events: none; }
        .thresh-rsasa { position: absolute; bottom: calc(120px + 26px + 8px + 18px); left: 0; right: 0; border-top: 1px dashed #94a3b8; opacity: 0.5; } /* 0.15 = 15% of 120px = 18px */
        .thresh-plddt { position: absolute; top: calc(120px + 26px + 8px + 84px); left: 0; right: 0; border-top: 1px solid #ef4444; opacity: 0.5; }   /* 70% of 120px = 84px */
        
        /* Masked IN */
        .col.in .rsasa-bar { background: #0ea5e9; }
        .col.in .seq-cell { background: #3b82f6; color: #ffffff; box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }
        .col.in .plddt-bar { background: #f59e0b; }
        
        /* Masked OUT */
        .col.out .rsasa-bar { background: #475569; opacity: 0.3; }
        .col.out .seq-cell { background: #334155; color: #64748b; }
        .col.out .plddt-bar { background: #475569; opacity: 0.3; }
        
        /* Tooltip */
        .tooltip { position: fixed; top: 20px; right: 20px; width: 250px; background: rgba(15, 23, 42, 0.95); border: 1px solid #475569; border-radius: 8px; padding: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: none; backdrop-filter: blur(8px); z-index: 100; pointer-events: none; }
        .tooltip h2 { margin: 0 0 12px 0; font-size: 16px; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 8px; display: flex; justify-content: space-between; }
        .tooltip .row { display: flex; justify-content: space-between; margin: 6px 0; font-size: 14px; }
        .tooltip .label { color: #94a3b8; }
        .tooltip .value { font-weight: 600; color: #f8fafc; font-family: monospace; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .badge.in { background: #166534; color: #4ade80; }
        .badge.out { background: #7f1d1d; color: #f87171; }
    </style>
</head>
<body>

    <h1>Target Fingerprint: {{ TARGET_ID }}</h1>
    <p class="subtitle">Hover over residues to view detailed track values and masking status.</p>

    <div class="chart-wrapper">
        <div class="chart-container" id="chart">
            <div class="lines-container">
                <div class="thresh-rsasa"></div>
                <div class="thresh-plddt"></div>
            </div>
            <!-- Columns injected by JS -->
        </div>
    </div>

    <div class="tooltip" id="tooltip">
        <h2><span>Position <span id="tt-pos"></span></span> <span id="tt-aa"></span></h2>
        <div class="row"><span class="label">Status</span> <span id="tt-status" class="badge"></span></div>
        <div class="row"><span class="label">rSASA</span> <span id="tt-rsasa" class="value"></span></div>
        <div class="row"><span class="label">pLDDT</span> <span id="tt-plddt" class="value"></span></div>
    </div>

    <script>
        const DATA = {{ JSON_DATA }};
        const chart = document.getElementById('chart');
        const tooltip = document.getElementById('tooltip');
        
        const MAX_RSASA_Y = 120; // 120px max height
        const MAX_PLDDT_Y = 120; // 120px max height
        
        for (let i = 0; i < DATA.sequence.length; i++) {
            const aa = DATA.sequence[i];
            const rsasa = DATA.rsasa[i];
            const plddt = DATA.plddt[i];
            const isMaskedIn = DATA.mask[i];
            const pos = DATA.positions[i];
            
            // Calculate heights
            // rSASA is 0.0 to ~1.0. Cap at 1.0 for visuals
            const rsasaHeight = Math.min(rsasa, 1.0) * MAX_RSASA_Y;
            // pLDDT is 0 to 100
            const plddtHeight = (plddt / 100) * MAX_PLDDT_Y;
            
            const col = document.createElement('div');
            col.className = 'col ' + (isMaskedIn ? 'in' : 'out');
            
            col.innerHTML = `
                <div class="hover-target"></div>
                <div class="rsasa-cell"><div class="rsasa-bar" style="height: ${rsasaHeight}px;"></div></div>
                <div class="seq-cell">${aa}</div>
                <div class="plddt-cell"><div class="plddt-bar" style="height: ${plddtHeight}px;"></div></div>
            `;
            
            // Hover logic
            const hoverTarget = col.querySelector('.hover-target');
            hoverTarget.addEventListener('mouseenter', () => {
                document.getElementById('tt-pos').textContent = pos;
                document.getElementById('tt-aa').textContent = aa;
                document.getElementById('tt-rsasa').textContent = rsasa.toFixed(3);
                document.getElementById('tt-plddt').textContent = plddt.toFixed(1);
                
                const statusEl = document.getElementById('tt-status');
                statusEl.textContent = isMaskedIn ? 'KEPT' : 'SKIPPED';
                statusEl.className = 'badge ' + (isMaskedIn ? 'in' : 'out');
                
                tooltip.style.display = 'block';
            });
            hoverTarget.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });
            
            chart.appendChild(col);
        }
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not args.input.exists():
        logger.error(f"Input LMDB not found: {args.input}")
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
    
    # 2. Get Boolean Mask
    mask_np = get_fingerprint_mask(
        sequence=target.sequence,
        sasa=target.sasa,
        plddt=target.residue_plddt,
        max_len=157
    )
    
    if mask_np is None:
        logger.warning(
            f"Target '{args.target_id}' fingerprint is shorter than 15 valid positions. "
            f"Highlighting none."
        )
        mask_list = [False] * len(target.sequence)
    else:
        mask_list = mask_np.tolist()

    # 3. Create JSON payload
    data_payload = {
        "sequence": target.sequence,
        "positions": target.position_ids,
        "rsasa": rsasa_np.tolist(),
        "plddt": target.residue_plddt,
        "mask": mask_list,
    }

    logger.info("Injecting data into HTML template...")
    html_content = HTML_TEMPLATE.replace("{{ TARGET_ID }}", args.target_id)
    html_content = html_content.replace("{{ JSON_DATA }}", json.dumps(data_payload))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Successfully generated visualization: {out_file}")


if __name__ == "__main__":
    main()
