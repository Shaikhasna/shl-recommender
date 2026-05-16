"""
download_catalog.py
Run this ONCE to download the SHL catalog and save it locally.
Usage: python download_catalog.py
"""

import json
import urllib.request
from pathlib import Path

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = Path("data/catalog.json")


def download():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Downloading catalog from:\n  {CATALOG_URL}\n")
    with urllib.request.urlopen(CATALOG_URL) as response:
        raw_bytes = response.read()
        raw = raw_bytes.decode("utf-8", errors="replace")
        raw = "".join(ch for ch in raw if ch >= " " or ch in "\n\r\t")
        data = json.loads(raw, strict=False)

    # Quick summary
    if isinstance(data, list):
        print(f"✅ Loaded {len(data)} assessments (list format)")
    elif isinstance(data, dict):
        for key in ("products", "data", "items", "catalog"):
            if key in data and isinstance(data[key], list):
                print(f"✅ Loaded {len(data[key])} assessments (key: '{key}')")
                break
        else:
            print(f"✅ Loaded dict with keys: {list(data.keys())}")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {OUTPUT_PATH}")
    print("\nFirst item sample:")
    first = data[0] if isinstance(data, list) else list(data.values())[0]
    if isinstance(first, list):
        first = first[0]
    print(json.dumps(first, indent=2)[:600])


if __name__ == "__main__":
    download()