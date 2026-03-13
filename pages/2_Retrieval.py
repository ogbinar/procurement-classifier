import streamlit as st

from utils_retrieval import load_retrieval_assets, retrieve_candidates

st.set_page_config(page_title="UNSPSC Retrieval", page_icon="🔎", layout="wide")
st.title("🔎 UNSPSC Retrieval")

@st.cache_resource
def cached_assets():
    return load_retrieval_assets()

try:
    bm25, bm25_ids, meta, collection, model = cached_assets()
except Exception as e:
    st.error(f"Failed to load retrieval assets: {e}")
    st.stop()

default_query = st.session_state.get("last_structured_description", "")

query = st.text_input(
    "Search query",
    value=default_query,
    placeholder="e.g. Mongol pencil, standard, for office writing",
)

topk_fused = st.slider("Number of fused results", min_value=3, max_value=20, value=5)

if st.button("Retrieve", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            normalized_query, df = retrieve_candidates(
                query=query,
                bm25=bm25,
                bm25_ids=bm25_ids,
                meta=meta,
                collection=collection,
                model=model,
                topk_fused=topk_fused,
            )

            st.subheader("Normalized query")
            st.code(normalized_query)

            st.subheader("Top results")
            if df.empty:
                st.warning("No results found.")
            else:
                st.dataframe(df, use_container_width=True)

                st.subheader("Detailed view")
                for _, row in df.iterrows():
                    with st.expander(f"#{row['rank']} - {row['code']} | {row['description']}"):
                        st.write(f"**Fused score:** {row['fused_score']:.6f}")
                        st.write(f"**BM25 rank:** {row['bm25_rank']}")
                        st.write(f"**Vector rank:** {row['vec_rank']}")
                        st.write(f"**Segment:** {row['segment']}")
                        st.write(f"**Family:** {row['family']}")
                        st.write(f"**Class:** {row['class_name']}")
                        st.write(f"**Commodity:** {row['commodity']}")
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
else:
    st.info("Enter a query and click Retrieve.")