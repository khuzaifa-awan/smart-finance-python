#!/usr/bin/env python3
import json
import base64
from pathlib import Path
import requests
import sys
from datetime import datetime

# ----------------------
# CONFIG (edit these values as needed)
# ----------------------
BASE_URL = "http://127.0.0.1:8000"   # API base URL
OUT_DIR = "stock_output"            # directory to save stock charts and gemini outputs

# Savings prediction payload (define your real user profile here)
PREDICT_PAYLOAD = {
    "income": 40000,
    "total_expenses": 32000,
    "goal_target_amount": 150000,
    "months_remaining": 24,
    "monthly_saving_label_raw": 2000,
    "expense_ratio": 0.8,
    "savings_rate": 0.05,
    "financial_health_score": 0.2,
    "total_potential_savings": 5000,
    "suggested_monthly_plan": 2500,
    "savings_is_synthetic": 0,
    "age": 30,
    "dependents": 2,
    "occupation": "Professional",
    "city_tier": "Tier_2",
    "education": 16,
    "desired_savings_percentage": 15,
    "potential_savings_education": 500
}

# Stock request params (define ticker & date range here)
STOCK_TICKER = "AAPL"
STOCK_START = "2024-01-01"
STOCK_END = "2024-06-30"

# ----------------------
# Helpers
# ----------------------
def pretty_print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def save_base64_image(b64str, out_path: Path):
    try:
        b = base64.b64decode(b64str)
        out_path.write_bytes(b)
        print(f"Saved image -> {out_path}")
    except Exception as e:
        print(f"Failed saving image {out_path}: {e}")

def timestamp_filename(prefix: str = "output") -> str:
    """Return a safe timestamped filename prefix."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # sanitize prefix
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("_", "-")).rstrip()
    return f"{ts}_{safe_prefix}.txt"

def save_gemini_text(text: str, label: str = "gemini") -> Path:
    """
    Save Gemini output text into OUT_DIR/gemini_outputs with a timestamped filename.
    Returns the Path to the saved file.
    """
    out_base = Path(OUT_DIR) / "gemini_outputs"
    out_base.mkdir(parents=True, exist_ok=True)
    fname = timestamp_filename(label)
    out_file = out_base / fname
    try:
        # Write with utf-8, preserve newlines
        out_file.write_text(text or "", encoding="utf-8")
        print(f"Saved Gemini text -> {out_file}")
    except Exception as e:
        print(f"Failed to save Gemini text to {out_file}: {e}")
    return out_file

# ----------------------
# Calls
# ----------------------
def call_predict_savings():
    url = f"{BASE_URL.rstrip('/')}/predict_savings"
    print(f"\nPOST {url}\nPayload:")
    pretty_print_json(PREDICT_PAYLOAD)
    try:
        r = requests.post(url, json=PREDICT_PAYLOAD, timeout=60)
        r.raise_for_status()
        resp = r.json()
        print("\n=== /predict_savings response ===")
        pretty_print_json(resp)

        # Save Gemini text if present
        gemini_text = resp.get("gemini_insights")
        if gemini_text:
            save_gemini_text(gemini_text, label="predict_savings")
        else:
            print("No 'gemini_insights' in predict_savings response to save.")

    except requests.exceptions.HTTPError as e:
        print("HTTP error calling /predict_savings:", e.response.status_code, e.response.text)
    except Exception as e:
        print("Error calling /predict_savings:", str(e))

def call_stock_insights():
    url = f"{BASE_URL.rstrip('/')}/stock_insights"
    payload = {
        "ticker": STOCK_TICKER,
        "start_date": STOCK_START,
        "end_date": STOCK_END
    }
    print(f"\nPOST {url}\nPayload:")
    pretty_print_json(payload)
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        print("\n=== /stock_insights response metadata ===")
        print("Ticker:", data.get("ticker"))
        print("\n=== Gemini Insights ===\n")
        print(data.get("gemini_insights", "No insights returned"))

        # Save Gemini text (use ticker label for filename)
        gemini_text = data.get("gemini_insights")
        if gemini_text:
            save_gemini_text(gemini_text, label=f"{STOCK_TICKER}_insights")
        else:
            print("No 'gemini_insights' in stock_insights response to save.")

        charts = data.get("charts", [])
        out_path = Path(OUT_DIR)
        out_path.mkdir(parents=True, exist_ok=True)

        for i, b64 in enumerate(charts, start=1):
            fname = out_path / f"{STOCK_TICKER}_chart_{i}.png"
            save_base64_image(b64, fname)

        print(f"\nSaved {len(charts)} chart(s) to {out_path.resolve()}")

    except requests.exceptions.HTTPError as e:
        print("HTTP error calling /stock_insights:", e.response.status_code, e.response.text)
    except Exception as e:
        print("Error calling /stock_insights:", str(e))

# ----------------------
# Main
# ----------------------
def main():
    print("Client starting. Base URL:", BASE_URL)
    call_predict_savings()
    call_stock_insights()
    print("\nClient finished.")

if __name__ == "__main__":
    main()
