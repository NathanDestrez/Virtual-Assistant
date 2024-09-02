from langchain.llms import OpenAI
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from function.utils_function import feedback
from streamlit_feedback import streamlit_feedback
import nltk

if "k" not in st.session_state:
    nltk.download('punkt')


def count_tokens(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Count the number of tokens
    num_tokens = len(tokens)
    return num_tokens


def _submit_feedback(user_feedback, emoji=None):
    st.toast(f"Feedback submitted: {user_feedback}", icon=emoji)
    return user_feedback


if "k" not in st.session_state:
    st.session_state.k = 0

st.set_page_config(page_title="Virtual assistant", page_icon="💬", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Virtual assistant 💬")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if ("k" in st.session_state) and ('messages' in st.session_state):
        st.session_state.k += len(st.session_state.messages)
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_engine' in st.session_state:
        del st.session_state.chat_engine

st.sidebar.markdown(
    "The virtual assistant is a web application to chat.")

with st.sidebar:
    st.markdown("---")
    st.warning("""
            :rotating_light: The virtual assistant can hallucinate. Consider checking important information.\n\n
            """)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


llm = OpenAI(base_url="http://192.168.48.32:1234/v1",
             openai_api_key="not-needed",
             temperature=1,
             top_p=0.05,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             streaming=True,
             )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for n, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message["role"] == "assistant" and n > 0:
        feedback_key = f"feedback_{n + st.session_state.k}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None

        user_response = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Please provide extra information",
            on_submit=_submit_feedback,
            key=feedback_key,
        )

        if user_response:
            app_name = "general"
            conversation = st.session_state.messages
            doc_content = ""
            if user_response['score'] == '👎':
                user_response['score'] = 0
            else:
                user_response['score'] = 1
            feedback(conversation, doc_content, app_name, user_response)

# Accept user input
if prompt := st.chat_input("What is up?"):
    if count_tokens(prompt) > 800:
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
