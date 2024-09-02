#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import json
import torch
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

torch.cuda.empty_cache()
print("clear")
rerun = False


def get_formatted_sources(doc):
    # Get the source and documentation path from the doc's metadata
    source = doc.metadata.get('source', 'Unknown Source')
    documentation = doc.metadata.get('documentation', 'Not Available')

    # Format the source as a Markdown link
    # Note: Streamlit doesn't currently support opening links in new tabs directly in Markdown.
    # Users will need to right-click and choose to open in a new tab.
    source_link = f"[{documentation}]({source})" if source != 'Unknown Source' else "Not Available"

    return source_link


# Load available collections from a JSON file
def load_collections(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data.get('collections', [])


def initialize_chroma(collection_name):
    try:
        chroma_client = chromadb.PersistentClient(path='C:/Users/Nathan/Desktop/RAG/database/v8')
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        langchain_chroma = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        return chroma_client, langchain_chroma
    except Exception as e:
        print(f"Error initializing Chroma: {e}")
        return None, None


# # Streamlit

# Mapping of display names to their corresponding keys
options_map = {
    "QMS": "QMS",
    "Skyminer": "Skyminer",
    "EPOCH-T": "EPOCH-T",
    "General": "General",
    "Python": "Python",
    "Translate_en_fr": "Translate_en_fr",
    "Translate_fr_en": "Translate_fr_en"
}

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if 'template' in st.session_state:
        del st.session_state.template
        rerun = True

st.sidebar.markdown("---")

current_selection = st.sidebar.selectbox(
    "Choose the collection you want to query:",
    options=list(options_map.keys()),
)

if current_selection == 'Python':
    max_tokens = 512
    top_p = 0.05
    temperature = 0.0
if current_selection == 'General':
    max_tokens = 256
    top_p = 0.5
    temperature = 0.7
else:
    max_tokens = 256
    top_p = 0.05
    temperature = 0.0

# max_tokens = st.sidebar.slider("max_tokens",
#                   min_value=1,
#                   max_value=512,
#                   value=max_tokens,
#                   help="max_token is the maximum length of the answer.")
# top_p = st.sidebar.slider("top_p",
#                   min_value=0.0,
#                   max_value=1.0,
#                   value=top_p,
#                   help="Close to 1: A large portion of the vocabulary is considered, making the output more random."
#                        " Close to 0: Only the very top tokens are considered, making the output more deterministic.")
# temperature = st.sidebar.slider("temperature",
#                   min_value=0.0,
#                   max_value=2.0,
#                   value=temperature,
#                   help="Lower value means more predictable output while higher allow for more creativity. ")


# Define your prompt templates
prompt_templates = {
    "Translate_en_fr": """
    [INST] Translate "{question}" in french [/INST]
    [INST] Write the translation only [/INST]
    {context} 
    """,

    "Translate_fr_en": """
    [INST] Translate "{question}" in english [/INST]
    [INST] Write the translation only [/INST]
    {context} 
    """,

    "General": """
    [INST] {question} [/INST]
    {context} 
    """,

    "Python": """
    [INST] Here the conversation with the user: {history} [/INST]
    [INST] You are a python expert. Your job is to help the user writing python code. [/INST]
    [INST] {question} [/INST]
    [INST] Explain what the task involves. [/INST]
    [INST] Generate the corresponding Python code in a code block. [/INST]
    {context}
    """,

    "QMS": """
    [INST] conversation = {history} [/INST]
    [INST] documentation = {context} [/INST]
    [INST] You are an expert to answer question about the documentation. [/INST]
    [INST] If you don't know the answer respond you have insufficient data to provide a specific answer. [/INST]
    [INST] If you don't understand the question respond you don't understand the question. [/INST]
    [INST] Utilize information from the documentation and the conversation to answer the question: {question}[/INST] 
    """,

    "Skyminer": """
    [INST] conversation = {history} [/INST]
    [INST] documentation = {context} [/INST]
    [INST] You are an expert to answer question about Skyminer, a data analytics tool. [/INST]
    [INST] In documentation you have all the documentation of the tool. [/INST]
    [INST] If you don't know the answer respond you have insufficient data to provide a specific answer. [/INST]
    [INST] If you don't understand the question respond you don't understand the question. [/INST]
    [INST] Utilize information from the documentation and the conversation to answer the question: {question}[/INST] 
    """,

    "EPOCH-T": """
    [INST] conversation = {history} [/INST]
    [INST] documentation = {context} [/INST]
    [INST] You are an expert to answer question about the documentation. [/INST]
    [INST] If you don't know the answer respond you have insufficient data to provide a specific answer. [/INST]
    [INST] If you don't understand the question respond you don't understand the question. [/INST]
    [INST] Utilize information from the documentation and the conversation to answer the question: {question}[/INST]    
    """
}

if ('template' not in st.session_state) or (st.session_state.selected_collection != current_selection):
    st.session_state.selected_collection = current_selection
    st.session_state.template = prompt_templates[options_map[current_selection]]
    # Initialize the memory for PromptTemplate
    if current_selection in ["Translate_fr_en", "Translate_fr_en", "General"]:
        st.session_state.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=st.session_state.template,
        )
    else:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )
    # Initialize the memory for conversation history
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )
    # Initialize the chat history
    st.session_state.chat_history = []
    # Initialize the chroma collection
    st.session_state.chroma_client, st.session_state.langchain_chroma = initialize_chroma(current_selection)
    # Initialize the retriever
    st.session_state.retriever = st.session_state.langchain_chroma.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2, "k": 5})
    # https://github.com/langchain-ai/langchain/blob/04caf07dee2e2843ab720e5b8f0c0e83d0b86a3e/libs/partners/openai/langchain_openai/llms/base.py#L97
    st.session_state.llm = OpenAI(base_url="http://192.168.48.33:1234/v1",
                                  openai_api_key="not-needed",
                                  temperature=temperature,
                                  top_p=top_p,
                                  max_tokens=max_tokens,
                                  presence_penalty=1.2,
                                  frequency_penalty=1.2,
                                  streaming=False,
                                  )

if rerun:
    rerun = False
    st.rerun()

st.title("Kratos Chatbot")

# Collection selection box
st.sidebar.markdown(f"There are : {st.session_state.langchain_chroma._collection.count()} items in the collection")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "The virtual assistant is a web application that allows you to chat "
    "with the documentation of your choice. The APP is developed to "
    "only answer question based on the documentation."
)

st.sidebar.markdown("__Other interactions have been limited__")
st.sidebar.markdown("---")
st.sidebar.markdown("The virtual assistant can make mistakes. Consider checking important information.")
st.sidebar.markdown("---")
st.sidebar.warning("""
    :rotating_light: The virtual assistant communicates exclusively in English, and attempting to converse with it in 
    French may result in nonsensical responses.
    """)
if current_selection == "EPOCH-T":
    st.sidebar.markdown("---")
    st.sidebar.warning(
        ":rotating_light: When documentation is on shackleton "
        "copy/past the path in your browser to access the documentation.\n\n "
        ":warning: The path might not work if you don't have the required access"
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
        if "sources" in message.keys():
            st.markdown(message["sources"], unsafe_allow_html=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=st.session_state.llm,
    chain_type='stuff',
    retriever=st.session_state.retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": st.session_state.prompt,
        "memory": st.session_state.memory,
    }
)

# After getting a response from the QA chain
if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Assistant is typing..."):
        response = qa_chain(user_input)

    retrieved_docs = st.session_state.retriever.get_relevant_documents(user_input)

    if current_selection in ['Skyminer', 'QMS', 'EPOCH-T']:
        # Start the sources markdown with a header
        sources_markdown = "##### Sources:\n\n"
        for i, doc in enumerate(retrieved_docs):
            documentation = doc.metadata.get('documentation', 'Not Available')
            source = doc.metadata.get('source', 'Unknown Source')
            path = doc.metadata.get('file_path', 'Not Available')
            # Format each source as an individual clickable link
            if path.startswith("http://") or source.startswith("https://"):
                # Enclose the link in HTML anchor tags to make it clickable
                source_link = f"<a href='{path}' target='_blank'>{source if source != 'Not Available' else source}</a>"
            else:
                # If it's not a valid URL, just show the text
                source_link = f"Path: **{path}**"
                # source_link = f"{documentation} - {path}"
            # Add each source link on a new line
            sources_markdown += f"{i + 1}. {source_link}<br>\n"

        chatbot_message = {
            "role": "assistant",
            "message": response['result'],
            "sources": sources_markdown
        }
    else:
        chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)
    st.rerun()
else:
    if current_selection == 'Translate_en_fr':
        st.write(f"Hi. I translate french into english.")
    elif current_selection == 'Translate_fr_en':
        st.write(f"Hi. I translate english into french.")
    elif current_selection == 'Python':
        st.write(f"Hi. I'm here to help you to write python code.")
    elif current_selection == 'General':
        st.write(f"Hi. I'm here to help. Feel free to ask me anything !")
    else:
        st.write(f"Hi. I'm here to help. Feel free to ask me anything related to {current_selection}")
