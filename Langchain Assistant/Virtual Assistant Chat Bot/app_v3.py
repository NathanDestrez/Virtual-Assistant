#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import json
import os
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Chroma
import chromadb 
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

import time
from IPython.display import display, HTML, clear_output
from streamlit_js_eval import streamlit_js_eval
import base64

# # Function

# In[12]:
torch.cuda.empty_cache()
print("clear")

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

available_collections = load_collections('Path/collections.json')


def initialize_chroma(collection_name):
    try:
        chroma_client = chromadb.PersistentClient(path='Path')
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



@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained('Path', device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained('Path', padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

def initialize_hf_pipeline():
    instruct_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float32,
        return_full_text=True,
        max_new_tokens=200,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
    return hf_pipe


# # Streamlit

# Mapping of display names to their corresponding keys
options_map = {
    "PRODUCT1": "Product1",
    "PRODUCT2": "Product2",
    "PRODUCT3": "Product3",
    "PRODUCT4": "Product4",
    "PRODUCT5": "Product5"
}

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
   if 'template' in st.session_state:
      del st.session_state.template
      current_selection = st.sidebar.selectbox(
        "Choose the collection you want to query:",
         options=list(options_map.keys()),
            )
      selected_collection = current_selection  
      st.rerun()

st.sidebar.markdown("---")

current_selection = st.sidebar.selectbox(
    "Choose the collection you want to query:",
     options=list(options_map.keys()),
        )
selected_collection = current_selection    

# Define your prompt templates
prompt_templates = {

    "General": """
    
    chat history = {history}
    ignore {context}
    Answer the following question with your own knowledge.
    Question: {question}

    You are a friendly assistant that answer general question. You can also simply  converse with the user.
    Be friendly, you can use humor. Your mission is to entertain the user. 
    
   
 
    Response: 
    """,

    "PRODUCT1": """
    ignore {context}
    chat history = {history}

    Write Python code with explanations, for the following requirement:
    Question: {question}

    Response:
   

    """,

    "PRODUCT2": """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    Instruction:
    You are an assistant to answer question about the Quality Management system documentation. 
    Use only information in the following paragraphs and the conversation history to answer the question at the end.
    Explain the answer with reference to these paragraphs and the history.
    If you don't have the information in paragraph or in the history then give response "Insufficient data to provide a specific answer." And stop the answer.

    {context}
    Chat history = {history}
    Question: {question}
    Response:
    """,

    "PRODUCT3": """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
    Instruction:
    You are an assistant to answer question about system in Product1 documentation.
    Use only information in the following paragraphs and the conversation history to answer the question at the end.
    Explain the answer with reference to these paragraphs and the history.
    If you don't have the information in paragraph or in the history then give response "Insufficient data to provide a specific answer".

    {context}
    Chat history = {history}
    Question: {question}
    Response:
    """,

    "PRODUCT4": """
         Below is an instruction that describes a task. Write a response that appropriately completes the request.
    Instruction:
    You are an assistant to answer question about system in Product2 documentation.
    Use only information in the following paragraphs and the conversation history to answer the question at the end.
    Explain the answer with reference to these paragraphs and the history.
    If you don't have the information in paragraph or in the history then give response "Insufficient data to provide a specific answer."

    {context}
    Chat history = {history}
    Question: {question}
    Response:
    """
}

if 'template' not in st.session_state or st.session_state.selected_collection != selected_collection:
    st.session_state.selected_collection = selected_collection
    st.session_state.template = prompt_templates[options_map[selected_collection]]
    # Initialize the memory for PromptTemplate
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
    # Check if the pipeline exists in the session state
    st.session_state.hf_pipe = initialize_hf_pipeline()
    # Initialize the chat history
    st.session_state.chat_history = [] 
    # Initialize the chroma collection
    st.session_state.chroma_client, st.session_state.langchain_chroma = initialize_chroma(selected_collection)
    # Initialize the retriever
    st.session_state.retriever = st.session_state.langchain_chroma.as_retriever(search_type='similarity', k=5)

st.title("Chatbot")

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
    :rotating_light: The virtual assistant communicates exclusively in English, and attempting to converse with it in French may result in nonsensical responses.
    """)
if selected_collection == "PRODUCTX": 
    st.sidebar.markdown("---") 
    st.sidebar.warning(
        ":rotating_light: When documentation is on shackleton "
        "copy/past the path in your browser to access the documentation.\n\n "
        ":warning: The path might not work if you don't have the requiered acess"
    )
        
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
        if "sources" in message.keys():
            st.markdown(message["sources"], unsafe_allow_html=True)
        
qa_chain = RetrievalQA.from_chain_type(
    llm=st.session_state.hf_pipe,
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
    
    retrieved_docs = st.session_state.langchain_chroma.similarity_search_with_relevance_scores(user_input, k=5)

    if selected_collection not in ['General', 'Python']:
        # Start the sources markdown with a header
        sources_markdown = "##### Sources:\n\n"
        for i, (doc, score) in enumerate(retrieved_docs):
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
                #source_link = f"{documentation} - {path}"
            # Add each source link on a new line
            sources_markdown += f"{i+1}. {source_link}<br>\n" 
    
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
   st.write(f"Hi. I'm here to help. Feel free to ask me anything related to {selected_collection}")




