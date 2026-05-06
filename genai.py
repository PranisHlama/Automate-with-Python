import streamlit as st
import ollama

# PAGE CONFIG
st.set_page_config(page_title="Pranish HRMS System")
st.title("Pranish HRMS System")

# System Prompt
SYSTEM_PROMPT = """
You are a professional customer support assistant for Odoo HRMS system
Rules:
- Answer clearly and politely.
- Be short and helpful.
- If unsure, say the issue will be forwarded to customer service.
"""

# SESSION MEMORY
if "messages" not in st.session_state:
    st.session_state.messages=[]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# USER INPUT
if prompt := st.chat_input("Ask your question here..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Streaming Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        MAX_HISTORY = 4
        recent_messages = st.session_state.messages[-MAX_HISTORY:]

        stream = ollama.chat(
            model="gemma",
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *recent_messages
            ],
            stream=True,
            options={
                "num_predict": 150,
                "temperature": 0.3
            }
        )


        for chunk in stream:
            full_response += chunk["message"]["content"]
            message_placeholder.markdown(full_response + "|")

        message_placeholder.markdown(full_response)

    # Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})


