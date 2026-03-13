import streamlit as st

st.set_page_config(
    page_title="SMC Procurement Helper",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 SMC Procurement Helper")
st.markdown(
    """
This app has two pages:

1. **Description Recommendation**  
   Standardize raw purchase request text into structured fields and a cleaner final description.

2. **UNSPSC Retrieval**  
   Retrieve likely UNSPSC matches using BM25 + Chroma + RRF fusion.
"""
)

st.info("Use the sidebar to switch between pages.")