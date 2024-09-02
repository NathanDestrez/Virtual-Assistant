import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from langchain_openai import OpenAI
from llama_index import SimpleDirectoryReader
from langchain.embeddings import SentenceTransformerEmbeddings
import pickle
import os
from function.utils_function import rating
from streamlit_star_rating import st_star_rating

if 'control' not in st.session_state:
    st.session_state.control = 0

local_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

llm = OpenAI(base_url="http://192.168.48.33:1234/v1",
             openai_api_key="not-needed",
             temperature=0.1,
             top_p=0.1,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             )

system_prompt = """
                You are an expert on skyminer and your job is to answer technical 
                questions. Assume that all questions are related to skyminer.
                Write your answer in markdown format.
                Keep your answers brief, clear, technical and based on facts – do not hallucinate features. \n\n
                """

st.set_page_config(page_title="Skyminer virtual assistant", page_icon="📈", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Skyminer virtual assistant 💬📈")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_engine' in st.session_state:
        del st.session_state.chat_engine
    if 'control' in st.session_state:
        st.session_state.control = 0

st.sidebar.markdown(
    "The Skyminer virtual assistant is a web application that allows you to chat "
    "with the skyminer documentation.")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Skyminer"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the kratos docs – hang tight! This should take 1-2 minutes."):
        if os.listdir("./data"):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            for doc in docs:
                if doc.metadata['file_path'] == 'data\\skyminer-development.pdf':
                    page = doc.metadata['page_label']
                    if page.isnumeric():
                        page = str(int(page) + 8)
                    elif page == 'i':
                        page = '3'
                    elif page == 'ii':
                        page = '4'
                    elif page == 'iii':
                        page = '5'
                    elif page == 'iv':
                        page = '6'
                    elif page == 'v':
                        page = '7'
                    elif page == 'vi':
                        page = '8'
                    doc.metadata['url'] = ("http://192.168.48.22:8082/repository/skyminer-dev/dev-env/skyminer"
                                           "-development.pdf#page=") + page
        else:
            docs = list()
        with open('docs.pkl', 'rb') as f:
            new_docs = pickle.load(f)
        docs += new_docs
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       embed_model=local_embeddings)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()
index.service_context.llm.system_prompt = system_prompt

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode='condense_question',
                                                        verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message.keys():
            st.markdown(message["sources"], unsafe_allow_html=True)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            answer = str()
            placeholder = st.empty()
            for token in response.response_gen:
                answer += token
                placeholder.markdown(answer)
            sources_markdown = "##### Sources:\n\n"
            for i, node in enumerate(response.source_nodes):
                if 'url' in node.metadata.keys():
                    url = node.metadata['url']
                    source_link = f"<a href='{url}' target='_blank'>{url.split('/')[-1]}</a>"
                elif 'file_path' in node.metadata.keys():
                    source_link = f"Path: **{node.metadata['file_path']}**"
                else:
                    continue
                sources_markdown += f"{i + 1}. {source_link}<br>\n"
            st.markdown(sources_markdown, unsafe_allow_html=True)
            if "sources_markdown" in globals():
                message = {"role": "assistant", "content": answer, "sources": sources_markdown}
            else:
                message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)  # Add response to message history
            st.session_state.control = 1

app_name = "general"
conversation = st.session_state.messages
if "response" in globals():
    doc_content = response.source_nodes
else:
    doc_content = ""

if st.session_state.control == 1:
    with st.sidebar:
        st.markdown("---")
        st.write("Please rate the answer:")
        rating_widget = st_star_rating(label="", maxValue=5, defaultValue=0, size=30)
        comment_widget = st.text_area(label="text_area", placeholder="Please enter your comment...",
                                      label_visibility="hidden")
    if st.sidebar.button("Submit"):
        rating(conversation, doc_content, app_name, rating_widget, comment_widget)
        st.session_state.control = 0
        st.rerun()
    with st.sidebar:
        st.sidebar.warning("""
                           :rotating_light: Please be aware that when you submit a rating,
                           the conversation you had with our application is saved for future reference or analysis.
                           """)

with st.sidebar:
    st.sidebar.markdown("---")
    st.sidebar.warning("""
                       :rotating_light: The APP is developed to only answer question based on the
                        Skyminer documentation, __other interactions have been limited__\n\n
                       :rotating_light: The virtual assistant can make mistakes. Consider checking important
                        information.\n\n
                       :rotating_light: The virtual assistant communicates exclusively in English, and attempting to
                       converse with it in French may result in nonsensical responses.\n\n
                       :rotating_light: When documentation is on shackleton copy/past the path in your browser to access
                       the documentation.\n\n
                       :warning: The path not work if you don't have the required access
                       """)
