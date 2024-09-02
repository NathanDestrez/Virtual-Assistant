from langchain.llms import OpenAI
import streamlit as st
from utils_function import display_message, StreamHandler, count_tokens
import nltk

if "k" not in st.session_state:
    nltk.download('punkt')
    st.session_state.k = 0

st.set_page_config(page_title="Virtual assistant", page_icon="💬", layout="centered",
                   initial_sidebar_state="expanded", menu_items=None)
st.title("Virtual assistant 💬")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if ("k" in st.session_state) and ('messages' in st.session_state):
        st.session_state.k += len(st.session_state.messages)
    if 'messages' in st.session_state:
        del st.session_state.messages

st.sidebar.markdown(
    "The virtual assistant is a web application to chat.")

with st.sidebar:
    st.markdown("---")
    st.warning("""
            :rotating_light: The virtual assistant can hallucinate. Consider checking important information.\n\n
            """)

llm = OpenAI(base_url="http://192.168.48.33:1234/v1",
             openai_api_key="not-needed",
             temperature=0.7,
             top_p=0.05,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             streaming=True,
             )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

display_message(st.session_state.messages, st.session_state.k, app_name='virtual_assistant', remove_space=False)

# Accept user input
if prompt := st.chat_input("What is up?"):
    if count_tokens(prompt) > 1000:
        st.warning(":warning: Your prompt is too long !")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Create history
        full_prompt = ""
        chat_history = st.session_state.messages.copy()
        if len(chat_history) > 3:
            chat_history = chat_history[-3:]
        for message in chat_history:
            full_prompt += message["role"] + ": " + message["content"] + "\n\n"
        full_prompt += f"assistant: "
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            response = llm(full_prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
