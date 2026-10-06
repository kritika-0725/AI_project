import streamlit as st

from core.graph import create_graph
from rag.pdf_loader import extract_pdf_text
from rag.retriever import DocumentRetriever


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NexaMind AI",
    page_icon="🧠",
    layout="wide",
)


# ==================================================
# LOAD CSS
# ==================================================

try:

    with open(
        "assets/style.css",
        "r",
        encoding="utf-8"
    ) as css_file:

        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )

except FileNotFoundError:

    pass


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "retriever" not in st.session_state:

    st.session_state.retriever = (
        DocumentRetriever()
    )


if "graph" not in st.session_state:

    st.session_state.graph = create_graph()


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="nexa-title">'
    '🧠 NexaMind AI'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nexa-subtitle">'
    'Your Intelligent AI Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ NexaMind AI")

    st.write(
        "Upload a PDF to ask questions "
        "about your documents."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
    )

    if uploaded_file:

        if st.button(
            "🔎 Process PDF",
            use_container_width=True,
        ):

            with st.spinner(
                "Reading and indexing PDF..."
            ):

                try:

                    pdf_bytes = (
                        uploaded_file.read()
                    )

                    text = extract_pdf_text(
                        pdf_bytes
                    )

                    if not text.strip():

                        st.error(
                            "No readable text was "
                            "found in this PDF."
                        )

                    else:

                        st.session_state.retriever.process_document(
                            text
                        )

                        st.session_state.pdf_name = (
                            uploaded_file.name
                        )

                        st.success(
                            "PDF processed successfully!"
                        )

                except Exception as error:

                    st.error(
                        f"Error: {error}"
                    )

    st.divider()

    if st.session_state.pdf_name:

        st.success(
            f"📚 {st.session_state.pdf_name}"
        )

    else:

        st.info(
            "💬 General AI mode"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask NexaMind anything..."
)


if question:

    # ------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # ------------------------------------------------
    # RETRIEVE DOCUMENT CONTEXT
    # ------------------------------------------------

    context = ""

    if st.session_state.retriever.ready:

        with st.spinner(
            "Searching your document..."
        ):

            try:

                context = (
                    st.session_state.retriever
                    .get_context(question)
                )

            except Exception as error:

                st.warning(
                    f"Document search failed: {error}"
                )


    # ------------------------------------------------
    # BUILD HISTORY
    # ------------------------------------------------

    history = []

    messages = st.session_state.messages

    for i in range(
        0,
        len(messages) - 1,
        2,
    ):

        if (
            messages[i]["role"] == "user"
            and messages[i + 1]["role"]
            == "assistant"
        ):

            history.append(
                {
                    "user": messages[i]["content"],
                    "assistant": messages[i + 1][
                        "content"
                    ],
                }
            )


    # ------------------------------------------------
    # LANGGRAPH
    # ------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "NexaMind is thinking..."
        ):

            result = (
                st.session_state.graph.invoke(
                    {
                        "question": question,
                        "context": context,
                        "history": history,
                        "answer": "",
                    }
                )
            )

            answer = result["answer"]

        st.markdown(answer)


    # ------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # ------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
