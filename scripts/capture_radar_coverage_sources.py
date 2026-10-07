"""Archive official candidate bytes; no QA, fact acceptance or model runs.

Each capture is immutable. Default mode verifies saved bytes; --fetch ID performs
one explicit network capture to ignored data/. A 200 HTML response to a PDF URL
is retained as failed access evidence, never admitted as a manual.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/radar_sources_v2/marine"
REPLACEMENT = ROOT / "data/radar_sources_v2/replacement"
URLS = {
    "furuno_nxt_entry": "https://www.furuno.com/files/Manual/158/upload/IME36680G3_DRS6A_12A_25A-NXT.pdf",
    "jrc_5200_entry": "https://www.jrc.co.jp/en/product/jma5200mk2",
    "garmin_fantom_manual": "https://static.garmin.com/pumac/GMR_Fantom_Open_Array_Install_Instructions_EN-US.pdf",
    "simrad_halo_entry": "https://www.simrad-yachting.com/en-eu/simrad/type/radar/halo-3-radar-w3-arrayri1220m-cbl/",
    "raymarine_cyclone_entry": "https://www.raymarine.com/en-gb/our-products/marine-radar/cyclone/cyclone-radar",
    "jrc_5200_catalog": "https://www.jrc.co.jp/hubfs/jrc-corp/assets/pdf/en/product/JMA-5200Mk2.pdf",
    "furuno_nxt_usa_manual": "https://furunousa.com/-/media/sites/furuno/document_library/documents/manuals/public_manuals/drs6anxt_12anxt_25anxt_installation_manual.pdf",
    "simrad_halo_2000_3000_manual": "https://softwaredownloads.navico.com/Simrad/SimradYachting_Software%20-%20Copy/Downloads/Radar/halo/HALO2000_3000_RADAR_IM_EN_988-12846-003_w.pdf",
    "raymarine_cyclone_manual_entry": "https://docs.raymarine.com/87402/en-US/latest/PhysicalSpecificationPedestalCyclon-BCB9E621.html",
    "raymarine_cyclone_spec_index": "https://docs.raymarine.com/87402/en-US/latest/TechnicalSpecificationCyclone-BA4412B1.html",
    "raymarine_cyclone_pro_spec_index": "https://docs.raymarine.com/87402/en-US/latest/TechnicalSpecificationCyclonePro-BA442F10.html",
    "metek_mrr_pro_entry": "https://metek.de/product/mrr-pro/",
    "metek_mrr_pro_20240705": "https://metek.de/wp-content/uploads/2016/12/2024-0705_MRR-PRO_datasheet_eng.pdf",
    "metek_mrr2_20230929": "https://metek.de/wp-content/uploads/2014/05/20230929_Datenblatt_MRR-2.pdf",
}
RAY_SECTIONS = {
    "antenna_physical": "PhysicalSpecificationAntennaCyclone-BCB9EEF6.html",
    "power": "PowerSpecification-BA4B0B00.html",
    "power_pro": "PowerSpecification-BA4C15BE.html",
    "environment": "EnvironmentalSpecificationCyclonePr-BCB9F5E5.html",
    "connections": "DataConnectionsCyclonePro-BCB9FA7B.html",
    "range": "RangeCyclonePro-BCBA0152.html",
    "range_pro": "Range-DD2CE309.html",
    "transmitter": "TransmitterSpecificationCyclonePro-BCBA0647.html",
    "transmitter_pro": "TransmitterSpecification-BF520DEC.html",
    "receiver": "RecieverSpecificationCyclonePro-BCBA0D7F.html",
    "antenna": "AntennaSpecificationCyclonePro-BCBA1613.html",
    "arpa": "ARPATargetTrackingCyclonePro-BCBA1B57.html",
}
URLS.update({"raymarine_spec_" + key: "https://docs.raymarine.com/87402/en-US/latest/" + value
             for key, value in RAY_SECTIONS.items()})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(key):
    target = (REPLACEMENT if key.startswith("metek_") else DEST) / key
    target.mkdir(parents=True, exist_ok=False)
    raw = target / "response.bin"
    url = URLS[key]
    result = subprocess.run([
        "/usr/bin/curl", "--location", "--silent", "--show-error",
        "--max-time", "45", "--max-filesize", "25000000", "--proto", "=https",
        "--proto-redir", "=https", "--user-agent", "Mozilla/5.0 (academic source archive)",
        "--output", str(raw), "--write-out", "%{json}", url,
    ], capture_output=True)
    info = json.loads(result.stdout) if result.stdout else {}
    data = raw.read_bytes() if raw.exists() else b""
    expected_pdf = ".pdf" in url.lower()
    kind = "pdf" if data.startswith(b"%PDF-") else "html" if "html" in info.get("content_type", "") else "other"
    status = "captured_candidate_bytes" if result.returncode == 0 and info.get("http_code") == 200 else "failed_http_capture"
    if status == "captured_candidate_bytes" and expected_pdf and kind != "pdf":
        status = "unexpected_content_not_a_pdf"
    meta = {
        "capture_id": key, "requested_url": url, "resolved_url": info.get("url_effective"),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "curl_exit_code": result.returncode, "http_status": info.get("http_code"),
        "content_type": info.get("content_type"), "detected_type": kind,
        "status": status, "file": str(raw.relative_to(ROOT)),
        "sha256": sha(data), "bytes": len(data),
        "error": result.stderr.decode(errors="replace")[:400],
        "human_verified": False, "evaluation_admitted": False,
    }
    (target / "capture.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def verify():
    rows = []
    for path in sorted([*DEST.glob("*/capture.json"), *REPLACEMENT.glob("*/capture.json")]):
        row = json.loads(path.read_text())
        raw = (ROOT / row["file"]).resolve()
        if not any(raw.is_relative_to(base.resolve()) for base in (DEST, REPLACEMENT)):
            raise ValueError("Capture outside local archive")
        data = raw.read_bytes()
        if sha(data) != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError(f"Capture bytes changed: {row['capture_id']}")
        rows.append({k: row[k] for k in ("capture_id", "status", "bytes", "detected_type")})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", choices=sorted(URLS))
    args = parser.parse_args()
    print(json.dumps(fetch(args.fetch) if args.fetch else verify(), ensure_ascii=False))
