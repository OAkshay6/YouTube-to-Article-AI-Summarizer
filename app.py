import streamlit as st
from utils import (
    get_smart_summarizer,
    generate_pdf,
    generate_website,
    extract_title
)

# ================================
# Page Config
# ================================

st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("YouTube to Article AI Summarizer")


# ================================
# Session State
# ================================

if "article" not in st.session_state:
    st.session_state.article = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "website_ready" not in st.session_state:
    st.session_state.website_ready = False


# ================================
# URL Input
# ================================

url = st.text_input("Enter YouTube URL")


# ================================
# Generate Article
# ================================

if st.button("Generate Article"):

    with st.spinner("Processing Video..."):

        article = get_smart_summarizer(url)

        st.session_state.article = article
        st.session_state.pdf_ready = False
        st.session_state.website_ready = False

    st.success("Article Generated")


# ================================
# Display Article
# ================================

if st.session_state.article:

    article = st.session_state.article

    st.markdown(article)


    # ================================
    # Generate PDF
    # ================================

    if st.button("Generate PDF"):

        video_title = extract_title(article)

        filename = generate_pdf(article, video_title)

        if filename:
            st.session_state.pdf_file = filename
            st.session_state.pdf_ready = True


if st.session_state.pdf_ready and st.session_state.pdf_file:

    with open(st.session_state.pdf_file, "rb") as f:

        st.download_button(
            "Download PDF",
            f,
            file_name=st.session_state.pdf_file
        )


    # ================================
    # Generate Website
    # ================================

    if st.button("Generate Website"):

        generate_website(article)

        st.session_state.website_file = "website.zip"
        st.session_state.website_ready = True


    if st.session_state.website_ready:

        with open(st.session_state.website_file, "rb") as f:

            st.download_button(
                "Download Website",
                f,
                file_name="website.zip"
            )