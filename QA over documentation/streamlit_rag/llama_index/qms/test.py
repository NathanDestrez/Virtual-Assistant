from langchain.llms import OpenAI
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from function.utils_function import *
from streamlit_feedback import streamlit_feedback


def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    return user_response


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


st.title("ChatGPT-like clone")

llm = OpenAI(base_url="http://192.168.48.32:1234/v1",
             openai_api_key="not-needed",
             temperature=0,
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
        feedback_key = f"feedback_{int(n)}"
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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_cb = StreamHandler(st.empty())
        response = llm(f'user: {prompt} \n assistant:', callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
