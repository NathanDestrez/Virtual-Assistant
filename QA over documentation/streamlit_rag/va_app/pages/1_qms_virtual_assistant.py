import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from langchain_openai import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
# noinspection PyUnresolvedReferences
from utils_function import feedback, count_tokens, StreamHandler, context_retriever, add_sources, display_message
import nltk
import pickle
from langchain.prompts import PromptTemplate

context_prompt_template = PromptTemplate.from_template("""
                  Conversation: {chat_history}  \n\n
                  context: question is about human resource.  \n\n
                  You are an assistant that reformulate question to a standalone question 
                  clearly, without typo and replacing  pronoun thanks to the context and the conversation.
                  Reformulate:
                  - question: {prompt}  \n\n
                  - standalone question: 
                  """)

full_prompt_template_fr = PromptTemplate.from_template("""
                Voici l'historique de la conversation :  \n\n
                {chat_history}  \n\n
                Voici la documentation :  \n\n
                {context}  \n\n
                Vous êtes un expert français sur le système de gestion de la qualité de Kratos.  \n\n
                Répondez à la question en tenant compte de la documentation et de l'historique de la conversation.  \n\n
                Si vous ne connaissez pas la réponse ou si vous ne comprenez pas la question, dites-le simplement.  \n\n
                Ne faites pas référence à la documentation dans votre réponse.  \n\n
                Question: {prompt}  \n\n
                Réponse:  
                """)

full_prompt_template_en = PromptTemplate.from_template("""
                Below is the conversation history:  \n\n
                {chat_history}  \n\n
                Context information is provided below:  \n\n
                {context}  \n\n
                You are an expert responding to questions about Kratos' Quality Management System.  \n\n
                Considering the context information and the conversation history,
                answer the question without prior knowledge.  \n\n
                If you don't know the answer or you don't understand the question just tell it.  \n\n
                Don't refer to the context in your answer.  \n\n                
                Question: {prompt}  \n\n
                Answer:  
                """)

if 'local_embeddings' not in st.session_state:
    st.session_state.local_embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2")

if "k" not in st.session_state:
    nltk.download('punkt')
    st.session_state.k = 0

llm = OpenAI(base_url="http://192.168.48.33:1234/v1",
             openai_api_key="not-needed",
             temperature=0,
             top_p=0.1,
             max_tokens=256,
             presence_penalty=1.2,
             frequency_penalty=1.2,
             streaming=True,
             )

st.set_page_config(page_title="QMS virtual assistant", page_icon="📃", layout="centered",
                   initial_sidebar_state="expanded", menu_items=None)

st.title("QMS virtual assistant 💬📃")

# New chat button
if st.sidebar.button(f"New chat", use_container_width=True, type="primary"):
    if ("k" in st.session_state) and ('messages' in st.session_state):
        st.session_state.k += len(st.session_state.messages)
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_engine' in st.session_state:
        del st.session_state.chat_engine

with st.sidebar:
    language = st.radio(
        "Language preferences:",
        ["FRA", "EN"],
        captions=["French (France)", "English United States"])
    st.markdown("---")
    st.markdown("""
        The QMS virtual assistant is a web application that enables you to ask questions about 
        QMS documentation or internal processes.  \n\n
        For each response, the virtual assistant will provide links to the sources it utilized.
        Each source is assigned a score between 0 and 1,  where 0 represents a document completely unrelated to the
        question, while 1 indicates a highly relevant document.
        """)
    st.markdown("---")
    st.warning("""
            :rotating_light: The virtual assistant can hallucinate. Consider checking important information.\n\n
            :rotating_light: The APP is developed to only answer question based on the
            QMS documentation and internal process, __other interactions have been limited__\n\n
            :rotating_light: Please be aware that when you submit a feedback,
            the conversation you had with our application is saved for future reference or analysis.
            """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Select prompt language
if language == "FRA":
    full_prompt_template = full_prompt_template_fr
else:
    full_prompt_template = full_prompt_template_en


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the QMS docs – hang tight! This should take 1-2 minutes."):
        with open('data/qms_docs.pkl', 'rb') as f:
            docs = pickle.load(f)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       chunk_size=512,
                                                       chunk_overlap=32,
                                                       embed_model=st.session_state.local_embeddings)
        indexer = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return indexer


index = load_data()

# Display chat messages from history on app rerun
display_message(st.session_state.messages, st.session_state.k, app_name='skyminer_virtual_assistant')

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
        chat_history = ""
        for message in st.session_state.messages[-3:-1]:
            chat_history += message["role"] + ": " + message["content"] + "\n\n"
        # Create context
        context_prompt = context_prompt_template.format(chat_history=chat_history, prompt=prompt)
        context, context_ = context_retriever(index, llm, prompt, context_prompt, max_sources=6, score_threshold=0.31)
        # Create the final prompt
        full_prompt = full_prompt_template.format(chat_history=chat_history, context=context, prompt=prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st_cb = StreamHandler(st.empty())
                response = llm(full_prompt, callbacks=[st_cb])
                message = add_sources(context_, response)
                st.session_state.messages.append(message)  # Add response to message history
                st.rerun()
