import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from langchain_openai import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from function.utils_function import rating
from streamlit_star_rating import st_star_rating

if 'control' not in st.session_state:
    st.session_state.control = 0

local_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OpenAI(base_url="http://192.168.48.33:1234/v1",
             openai_api_key="not-needed",
             temperature=1,
             top_p=0.5,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             )

system_prompt = """
                Write in good markdown format.
                You are a friendly human, that act like a human.
                """

st.set_page_config(page_title="Virtual assistant", page_icon="💬", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Virtual assistant 💬")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_engine' in st.session_state:
        del st.session_state.chat_engine
    if 'control' in st.session_state:
        st.session_state.control = 0

st.sidebar.markdown(
    "The virtual assistant is a web application to chat.")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the kratos docs – hang tight! This should take 1-2 minutes."):
        docs = list()
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       embed_model=local_embeddings)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()
index.service_context.llm.system_prompt = system_prompt

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode='simple', verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        rating_widget = st_star_rating("", maxValue=5, defaultValue=0, size=30)
        comment_widget = st.text_area(label="", placeholder="Please enter your comment...")
    if st.sidebar.button("Submit"):
        rating(conversation, doc_content, app_name, rating_widget, comment_widget)
        st.session_state.control = 0
        st.rerun()

with st.sidebar:
    st.markdown("---")
    st.warning("""
            :rotating_light: The virtual assistant can hallucinate. Consider checking important information.\n\n
            """)
