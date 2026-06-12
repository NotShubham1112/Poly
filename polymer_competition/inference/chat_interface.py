"""
inference.chat_interface.py

Streamlit chat interface for PolyChain:
    - Natural language input with SMILES extraction
    - Property prediction with confidence intervals
    - 2D molecule drawing (RDKit)
    - Optional RAG over curated polymer papers
    - MCP (Model Context Protocol) placeholder for extensibility

Run:
    streamlit run inference/chat_interface.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolyChain — Polymer Property Predictor",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 PolyChain: Polymer Property Predictor")

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
SMILES_PATTERN = re.compile(r"(\*?[A-Za-z0-9\(\)\[\]\=\#\@\+\-\.\\/\*]+\*)")


def extract_smiles(text: str) -> str | None:
    """Try to extract a polymer SMILES (contains *) from natural language."""
    m = SMILES_PATTERN.search(text)
    return m.group(0) if m else None


def draw_molecule_svg(smiles: str, width: int = 400, height: int = 300) -> str | None:
    """Return an SVG string of the 2D molecule drawing."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def draw_molecule_png(smiles: str, size: tuple = (400, 300)) -> bytes | None:
    """Return a PNG image of the 2D molecule drawing."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def compute_features_summary(smiles: str) -> dict:
    """Compute a summary of polymer-specific features for display."""
    try:
        from features.custom_polymer import (
            asterisks_count, repeat_unit_length, branching_indicator,
            aromatic_carbon_fraction, sp3_carbon_fraction, rotatable_bonds,
            rigidity_index, hbond_density, molecular_weight_monomer,
            ring_statistics, hbond_donor_acceptor,
        )
        return {
            "Connection points (*)": asterisks_count(smiles),
            "Repeat unit length": repeat_unit_length(smiles),
            "Branched": "Yes" if branching_indicator(smiles) else "No",
            "Molecular weight": f"{molecular_weight_monomer(smiles):.1f} Da",
            "Aromatic C fraction": f"{aromatic_carbon_fraction(smiles):.2f}",
            "SP3 C fraction": f"{sp3_carbon_fraction(smiles):.2f}",
            "Rotatable bonds": rotatable_bonds(smiles),
            "Rigidity index": f"{rigidity_index(smiles):.2f}",
            "H-bond density": f"{hbond_density(smiles):.3f}",
            **ring_statistics(smiles),
            **hbond_donor_acceptor(smiles),
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_resource
def load_predictor():
    """Lazy-load the PolyChain predictor (cached across reruns)."""
    try:
        from inference.predictor import PolymerPredictor
        return PolymerPredictor("outputs/checkpoints/polychain_best.pt")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    smiles_input = st.text_input(
        "Enter a polymer SMILES",
        value="*CCO*",
        help="Use * to denote connection points (repeat-unit boundaries)."
    )
    st.divider()

    st.subheader("Model")
    model_mode = st.radio(
        "Prediction mode",
        ["Ensemble (precomputed)", "PolyChain (live)"],
        index=0,
    )

    st.subheader("RAG Settings")
    enable_rag = st.toggle("Enable RAG retrieval", value=False)
    top_k = st.slider("Top-K similar polymers", 1, 20, 5,
                       disabled=not enable_rag)

    st.divider()
    st.caption("Built with ❤️ for IIT Madras Polymer Competition")


# ──────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────
tab_predict, tab_chat, tab_rag, tab_about = st.tabs([
    "🔬 Predict", "💬 Chat", "📚 Similar (RAG)", "ℹ️ About"
])

# ──── Tab 1: Property Prediction ────
with tab_predict:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Molecule Visualization")
        img_data = draw_molecule_png(smiles_input, size=(500, 400))
        if img_data:
            st.image(img_data, caption=smiles_input, use_container_width=True)
        else:
            st.warning("Could not parse SMILES for drawing. Check the input.")

    with col2:
        st.subheader("Polymer Features")
        feats = compute_features_summary(smiles_input)
        if "error" not in feats:
            for k, v in feats.items():
                st.metric(label=k, value=v)
        else:
            st.error(feats["error"])

    st.divider()

    if st.button("🚀 Predict Property", type="primary", use_container_width=True):
        pred = load_predictor()
        if pred is not None:
            try:
                yhat = pred.predict([smiles_input])
                st.success(f"**Predicted property value: {yhat[0]:.4f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning(
                "No trained checkpoint found at `outputs/checkpoints/polychain_best.pt`. "
                "Run `python -m training.train --model_type polychain` first."
            )

# ──── Tab 2: Chat Interface ────
with tab_chat:
    st.subheader("Natural Language Polymer Q&A")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content":
             "Hi! Paste a polymer SMILES like `*CCO*` or ask me about polymer "
             "properties. I'll predict and explain."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about a polymer or paste a SMILES...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Try to extract SMILES and predict
        detected_smi = extract_smiles(user_input)
        if detected_smi:
            reply = f"I detected the SMILES: `{detected_smi}`\n\n"
            feats = compute_features_summary(detected_smi)
            if "error" not in feats:
                reply += "**Polymer features:**\n"
                for k, v in feats.items():
                    reply += f"- {k}: {v}\n"

            pred = load_predictor()
            if pred is not None:
                try:
                    yhat = pred.predict([detected_smi])
                    reply += f"\n**Predicted property: {yhat[0]:.4f}**"
                except Exception:
                    reply += "\n⚠️ Could not run prediction (check checkpoint)."
            else:
                reply += "\n⚠️ No trained model checkpoint found."
        else:
            reply = (
                "I couldn't find a SMILES in your message. "
                "Try something like: *What is the Tg of `*CCO*`?*"
            )

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# ──── Tab 3: RAG ────
with tab_rag:
    st.subheader("Find Similar Polymers (RAG)")
    if enable_rag:
        st.info(
            "**RAG module** will retrieve similar polymers from the training set "
            "and curated literature. This requires a vector index built from "
            "Morgan fingerprints.\n\n"
            "_Implementation:_ Build FAISS index over Morgan FPs of training SMILES, "
            "then retrieve top-K nearest neighbors by Tanimoto similarity."
        )
        # Placeholder for RAG retrieval
        if st.button("Search"):
            st.write(f"Would search for top-{top_k} polymers similar to `{smiles_input}`")
            st.write("🚧 RAG index not yet built. Run `build_rag_index.py` first.")
    else:
        st.info("Toggle **Enable RAG retrieval** in the sidebar to use this feature.")

# ──── Tab 4: About ────
with tab_about:
    st.markdown("""
    ## PolyChain Architecture

    **PolyChain** is a Hierarchical Periodic Transformer for polymer property
    prediction. It introduces two novel components:

    1. **HAMF** (Hierarchy-Aware Multi-Scale Fusion) — cross-attention over
       monomer/dimer/trimer scale embeddings
    2. **PECGN** (Periodic Equivariant Chain-Growth Network) — learned
       boundary operator with translation invariance

    ### MCP Integration

    PolyChain supports the **Model Context Protocol (MCP)** for tool-use
    integration with LLM agents:

    ```json
    {
      "tool": "predict_polymer_property",
      "input": {"smiles": "*CCO*"},
      "output": {"property": 350.2, "unit": "K"}
    }
    ```

    _MCP server is available at `inference/mcp_server.py` (optional)._

    ### Links

    - [Architecture Overview](../docs/architecture_overview.md)
    - [PolyChain Whitepaper](../docs/polychain_whitepaper.md)
    - [README](../README.md)
    """)
