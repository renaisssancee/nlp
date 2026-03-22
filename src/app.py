"""Streamlit UI for CineMatch movie recommendation system."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.rag import CineMatchRAG

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="centered")

st.title("🎬 CineMatch")
st.caption("Рекомендательная система фильмов на основе RAG")


@st.cache_resource
def get_rag():
    """Initialize RAG pipeline (cached)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OPENROUTER_API_KEY не найден. Создайте файл .env с ключом.")
        st.stop()
    return CineMatchRAG(api_key)


rag = get_rag()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_ids" not in st.session_state:
    st.session_state.request_ids = {}

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show feedback buttons for assistant messages
        if msg["role"] == "assistant" and i in st.session_state.request_ids:
            req_id = st.session_state.request_ids[i]
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    rag.save_feedback(req_id, "like")
                    st.toast("Спасибо за отзыв!")
            with col2:
                if st.button("👎", key=f"dislike_{i}"):
                    rag.save_feedback(req_id, "dislike")
                    st.toast("Спасибо за отзыв!")

# Chat input
if user_input := st.chat_input("Опишите, какой фильм вы хотите посмотреть..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Ищу фильмы..."):
            # Build history for context
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude current
            ]
            try:
                result = rag.query(user_input, history)
            except Exception as e:
                error_msg = "Произошла ошибка при обработке запроса. Попробуйте ещё раз через несколько секунд."
                print(f"RAG query error: {e}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()

        # Format response
        if result["type"] == "off_topic":
            response_text = result["message"]
        elif result["type"] == "no_results":
            response_text = result["message"]
        else:
            # Format movie recommendations
            parts = []
            if result.get("message"):
                parts.append(result["message"])
            parts.append("")

            for j, movie in enumerate(result.get("movies", []), 1):
                title = movie.get("title", "Unknown")
                year = movie.get("year", "")
                rating = movie.get("rating", "")
                duration = movie.get("duration_min", "")
                reason = movie.get("reason", "")

                parts.append(
                    f"**{j}. {title}** ({year})\n"
                    f"   ⭐ {rating} | ⏱ {duration} мин\n"
                    f"   _{reason}_"
                )
                parts.append("")

            response_text = "\n".join(parts)

        st.markdown(response_text)

        # Store message and request_id
        msg_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        if result.get("request_id"):
            st.session_state.request_ids[msg_idx] = result["request_id"]

        # Feedback buttons
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"like_{msg_idx}"):
                rag.save_feedback(result["request_id"], "like")
                st.toast("Спасибо за отзыв!")
        with col2:
            if st.button("👎", key=f"dislike_{msg_idx}"):
                rag.save_feedback(result["request_id"], "dislike")
                st.toast("Спасибо за отзыв!")
