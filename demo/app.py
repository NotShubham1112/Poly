"""
demo.app.py
Standalone Streamlit entry point.

Run:
    streamlit run demo/app.py
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.predictor import PolymerPredictor


st.set_page_config(page_title="PolyChain Demo", page_icon="🧬")
st.title("🧬 PolyChain Demo")

smiles = st.text_input("Polymer SMILES", value="*CCO*")
if st.button("Predict"):
    try:
        pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")
        yhat = pred.predict([smiles])
        st.metric("Predicted Tg (°C)", f"{yhat[0]:.2f}")
    except Exception as e:
        st.error(str(e))
