import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from langchain_openai import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from function.utils_function import feedback, count_tokens, StreamHandler
from streamlit_feedback import streamlit_feedback
import nltk
import pickle


def _submit_feedback(user_feedback, emoji=None):
    st.toast(f"Feedback submitted: {user_feedback}", icon=emoji)
    return user_feedback


if 'local_embeddings' not in st.session_state:
    st.session_state.local_embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2")

if "k" not in st.session_state:
    nltk.download('punkt')
    st.session_state.k = 0

llm = OpenAI(base_url="http://192.168.48.32:1234/v1",
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
    st.markdown("""
        The QMS virtual assistant is a web application that enables you to ask questions about 
        QMS documentation or internal processes.  \n\n
        For each response, the virtual assistant will provide links to the sources it utilized.
        Each source is assigned a score between 0 and 1,  where 0 represents a document completely unrelated to the
        question, while 1 indicates a highly relevant document.
        """)
    st.markdown("---")
    language = st.radio(
        "Language preferences:",
        ["FRA", "EN"],
        captions=["French (France)", "English United States"])
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


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the kratos docs – hang tight! This should take 1-2 minutes."):
        with open('docs.pkl', 'rb') as f:
            docs = pickle.load(f)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       chunk_size=512,
                                                       chunk_overlap=32,
                                                       embed_model=st.session_state.local_embeddings)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()

# Display chat messages from history on app rerun
for n, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if "sources" in message.keys():
            st.markdown(message["sources"], unsafe_allow_html=True)
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
        # Create context
        context_prompt = f"""
            Conversation: {chat_history[:-1]}  \n\n
            Query: {prompt}  \n\n
            You are an assistant that generates a new query based on the Query in the human resource context.  \n\n
            Feel free to utilize the Conversation for assistance as necessary.  \n\n
            For example:  \n\n
            - If a pronoun in the query refers to a word in the provided conversation,
              replace the pronoun with the word.  \n\n
            - If there are typographical errors, correct them.  \n\n
            Generate a new query: 
            - new query:
            """
        query = llm(context_prompt)
        query = query.split('(')[0].split('?')[0]
        st.write(query)
        context_list = index.as_retriever(similarity_top_k=2).retrieve(query)
        context_list += index.as_retriever(similarity_top_k=2).retrieve(prompt)
        # Document selection
        context_ = list()
        score_list = list()
        # Remove duplicates
        for element in context_list:
            if element.text not in [ele.text for ele in context_]:
                context_.append(element)
                score_list.append(element.score)
        combined_list = list(zip(score_list, context_))
        combined_list.sort(key=lambda x: x[0], reverse=True)
        score_list, context_ = zip(*combined_list)
        print(score_list)
        print(context_)
        # Max 3 documents cause of 2048 token mistral limit
        if len(context_) > 3:
            context_ = context_[:3]
        # Remove bad documents
        indices_to_remove = [i for i, score in enumerate(score_list) if score < 0.3]
        for index in reversed(indices_to_remove):
            del score_list[index]
            del context_[index]
        sources = [f'\n\n - {el.text}' for el in context_list]
        print(f"Sources: {sources}")
        context = ""
        for n, doc in enumerate(context_):
            metadata_keys = list(set(doc.metadata) - set(doc.node.excluded_llm_metadata_keys))
            context += f"- "
            for key in metadata_keys:
                context += f'{key}: {doc.metadata[key]} \n '
            context += 'text: ' + doc.text + ' ] \n\n '

        # Display assistant response in chat message container
        if language == "FRA":
            full_prompt = (f"Voici l'historique de la conversation ci-dessous:  \n\n {chat_history[:-1]} "
                           f"\n ----------------- \n"
                           f"Les informations de contexte sont ci-dessous:  \n\n {context} "
                           f"\n ----------------- \n"
                           f"En tenant compte des informations de contexte et de l'historique de la conversation,"
                           f"sans connaissance préalable, répondez à la requête. \n"
                           f"Vous êtes un expert français qui répond aux questions sur le quality management"
                           f"system de Kratos. \n\n"
                           f"Requête : {prompt}\n\n"
                           f"Réponse : ")
        else:
            full_prompt = (f"Below is the conversation history:  \n\n {chat_history[:-1]} "
                           f"\n ----------------- \n"
                           f"Context information is provided below:  \n\n {context} "
                           f"\n ----------------- \n"
                           f"Considering the context information and the conversation history,"
                           f"answer the query without prior knowledge. \n"
                           f"You are an expert responding to questions about Kratos' Quality Management System. \n"
                           f"Query: {prompt}\n\n"
                           f"Response: ")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st_cb = StreamHandler(st.empty())
                response = llm(full_prompt, callbacks=[st_cb])
                sources_markdown = "##### Sources:\n\n"
                for i, node in enumerate(context_):
                    if 'url' in node.metadata.keys():
                        url = node.metadata['url']
                        if 'page_label' in node.metadata.keys():
                            source_link = (f"<a href='{url}' target='_blank'>{url.split('/')[-1]}</a> "
                                           f" / page: {node.metadata['page_label']}")
                        else:
                            source_link = f"<a href='{url}' target='_blank'>{url.split('/')[-1]}</a>"
                    elif 'file_path' in node.metadata.keys():
                        source_link = f"Path: **{node.metadata['file_path']}**"
                    else:
                        continue
                    source_link += f" / Score: {round(node.score, 2)}"
                    sources_markdown += f"{i + 1}. {source_link}<br>\n"
                st.markdown(sources_markdown, unsafe_allow_html=True)
                if "sources_markdown" in globals():
                    message = {"role": "assistant", "content": response, "sources": sources_markdown}
                else:
                    message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)  # Add response to message history
                st.rerun()
