"""
demo/app.py
Standalone Streamlit entry point (mirrors inference/chat_interface.py).

Run:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-export the full chat interface — identical functionality
# This file exists so `streamlit run demo/app.py` works as a standalone entry.
exec(open(str(Path(__file__).resolve().parent.parent / "inference" / "chat_interface.py")).read())
