import streamlit as st
import random
import time

st.write("I love LLMs")

st.caption("You can ask me questions that is in my database and I will be sure to answer.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages= [{"role": "assistant", "content": "Let's start our chat with a Hello?"}]

# Display chat messages from history on app return
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("What is up?"):

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response=""
        assistant_response = random.choice(
                [
                    "Hello there! How can I assist you today?",
                    "Hi, human! Is there anything I can help you with?",
                    "Do you need help?",
                ]
            )
    for chunk in assistant_response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})




