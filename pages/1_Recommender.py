import json
import streamlit as st

from utils_recommender import (
    assess_description_quality,
    build_description,
    call_ollama_structurer,
    choose_questions,
    missing_critical_fields,
)

st.set_page_config(page_title="Description Recommendation", page_icon="📝", layout="wide")
st.title("📝 Description Recommendation")

raw_text = st.text_area(
    "Raw item description",
    placeholder="e.g. pencil for office work use",
    height=140,
)

if raw_text.strip():
    quality = assess_description_quality(raw_text)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Quality")
        st.metric("Label", quality["label"])

    with col2:
        st.subheader("Issues")
        if quality["issues"]:
            for issue in quality["issues"]:
                st.write(f"- {issue}")
        else:
            st.success("No major issues detected.")

    questions = choose_questions(quality["label"], raw_text)

    st.subheader("Optional clarification inputs")
    answers = {
        "base_item": "",
        "variant": "",
        "spec": "",
        "use_case": "",
    }

    if questions:
        for field, question in questions:
            answers[field] = st.text_input(question, key=f"q_{field}")
    else:
        st.caption("No clarification questions suggested.")

    if st.button("Generate recommendation", type="primary"):
        structured = call_ollama_structurer(raw_text, answers)
        final_description = build_description(
            base_item=structured.get("base_item", ""),
            variant=structured.get("variant", ""),
            spec=structured.get("spec", ""),
            use_case=structured.get("use_case", ""),
        )
        missing = missing_critical_fields(structured)

        st.subheader("Structured fields")
        st.code(json.dumps(structured, indent=2, ensure_ascii=False), language="json")

        st.subheader("Final recommended description")
        if final_description:
            st.success(final_description)
        else:
            st.warning("No final description could be generated yet.")

        if missing:
            st.subheader("Still missing critical information")
            for field in missing:
                st.write(f"- {field}")

        st.session_state["last_structured_description"] = final_description
        st.session_state["last_structured_fields"] = structured

else:
    st.info("Enter a raw item description to begin.")