"""
inference.chat_interface.py

Streamlit chat interface: natural-language Q&A, molecule drawing, RAG over papers.

Run:
    streamlit run inference/chat_interface.py
"""
from __future__ import annotations

import streamlit as st

from .predictor import PolymerPredictor


st.set_page_config(page_title="PolyChain", page_icon="🧬", layout="wide")
st.title("🧬 PolyChain: Polymer Property Predictor")

# Lazy-load predictor
@st.cache_resource
def load_predictor():
    return PolymerPredictor("outputs/checkpoints/polychain_best.pt")


# Sidebar
with st.sidebar:
    st.header("Settings")
    smiles = st.text_input("Enter a polymer SMILES", value="*CCO*")
    top_k = st.slider("Top similar polymers (RAG)", min_value=1, max_value=20, value=5)

# Main panel
tab1, tab2, tab3 = st.tabs(["Predict", "Similar (RAG)", "About"])

with tab1:
    st.subheader("Predict a property")
    if st.button("Predict"):
        try:
            pred = load_predictor()
            yhat = pred.predict([smiles])
            st.metric(label="Predicted Tg (°C)", value=f"{yhat[0]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("Find similar polymers")
    st.info("RAG over PolyInfo + curated papers (not yet implemented).")

with tab3:
    st.markdown("""
    **PolyChain** – Hierarchical Periodic Transformer with Equivariant
    Multi-Scale Graph Reasoning for Polymer Property Prediction.

    See the project [README](../README.md) for architecture details.
    """)
