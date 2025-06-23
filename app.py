import os
import subprocess
import streamlit as st
from pipeline import process_url, summarization, ask_question
from retriever import Retriever

try:
    import sentence_transformers
except ImportError:
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

# Page setup
st.set_page_config(page_title="Anime Summarizer Chat", layout="wide")
st.title("Anime LLM Assistant")

# State initialization
if "page" not in st.session_state:
    st.session_state.page = "url_input"
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "url" not in st.session_state:
    st.session_state.url = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ========== Page 1: URL Input ==========
def page_url_input():
    st.header("Step 1: Enter a Fandom URL")
    st.session_state.url = st.text_input("Enter Anime Fandom URL:", value=st.session_state.url)

    if st.button("Process URL"):
        if st.session_state.url:
            with st.spinner("Scraping and summarizing..."):
                try:
                    paragraphs = process_url(st.session_state.url)
                    summary = summarization(paragraphs)
                    retriever = Retriever(paragraphs)

                    # Save to session state
                    st.session_state.paragraphs = paragraphs
                    st.session_state.summary = summary
                    st.session_state.retriever = retriever
                    st.session_state.chat_history = []
                    st.session_state.page = "chat"
                except Exception as e:
                    st.error(f"Error processing URL: {e}")

# ========== Page 2: Chat Interface ==========
def page_chat():
    st.header("Anime Summary")
    st.info(st.session_state.summary)

    st.divider()
    st.header("Ask a Question")
    user_input = st.text_input("Ask something about the anime:")

    if st.button("Ask") and user_input:
        try:
            response = ask_question(user_input, st.session_state.retriever)
            st.session_state.chat_history.append((user_input, response))
        except Exception as e:
            st.error(f"Error answering question: {e}")

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, r in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {r}")

    if st.button("Start Over"):
        for key in ["summary", "url", "chat_history", "retriever", "paragraphs", "page"]:
            st.session_state[key] = None
        st.session_state.page = "url_input"

# ========== Page Routing ==========
if st.session_state.page == "url_input":
    page_url_input()
elif st.session_state.page == "chat":
    page_chat()
