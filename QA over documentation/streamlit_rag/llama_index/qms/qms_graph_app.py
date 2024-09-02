import streamlit as st
from llama_index import KnowledgeGraphIndex, ServiceContext
from langchain_openai import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
import pickle
from function.utils_function import rating
from streamlit_star_rating import st_star_rating
from pyvis.network import Network

if 'control' not in st.session_state:
    st.session_state.control = 0

if 'local_embeddings' not in st.session_state:
    st.session_state.local_embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2")

llm = OpenAI(base_url="http://192.168.48.32:1234/v1",
             openai_api_key="not-needed",
             temperature=0,
             top_p=0.1,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             )

context_prompt = """
                You are an expert on documentation and your job is to answer technical questions. \n
                Here are the relevant documents for the context:\n
                {context_str}
                \nInstruction: Use the previous chat history, or the context above, to interact and help the user.
                Write your answer in good markdown format. \n
                Keep your answers brief, clear, technical and based on facts – do not hallucinate features. \n
                """

st.set_page_config(page_title="QMS virtual assistant", page_icon="📈", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("QMS virtual assistant 💬📈")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_engine' in st.session_state:
        del st.session_state.chat_engine
    if 'control' in st.session_state:
        st.session_state.control = 0

st.sidebar.markdown(
    "The QMS virtual assistant is a web application that allows you to chat "
    "with the QMS documentation.")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about documentation"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the kratos docs – hang tight! This should take 1-2 minutes."):
        with open('docs_pdf.pkl', 'rb') as f:
            docs = pickle.load(f)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       chunk_size=512,
                                                       context_window=1500,
                                                       chunk_overlap=32,
                                                       embed_model=st.session_state.local_embeddings)
        index = KnowledgeGraphIndex.from_documents(
            docs,
            max_triplets_per_chunk=2,
            service_context=service_context,
            include_embeddings=True,
        )
        g = index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show("example.html")
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_query_engine(
                                                include_text=True,
                                                response_mode="tree_summarize",
                                                embedding_mode="hybrid",
                                                similarity_top_k=5,
                                                verbose=True
                                                        )

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
            response = st.session_state.chat_engine.query(prompt)
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
                        QMS documentation, __other interactions have been limited__\n\n
                       :rotating_light: The virtual assistant can make mistakes. Consider checking important
                        information.\n\n
                       :rotating_light: The virtual assistant communicates exclusively in English, and attempting to
                       converse with it in French may result in nonsensical responses.\n\n
                       :rotating_light: When documentation is on shackleton copy/past the path in your browser to access
                       the documentation.\n\n
                       :warning: The path not work if you don't have the required access
                       """)
