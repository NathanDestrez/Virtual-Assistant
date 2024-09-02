#!/usr/bin/env python
# coding: utf-8

# # Import

# In[99]:


import re
import os
import csv
import datetime
from datetime import datetime
import pandas as pd

# vector store set up 

import chromadb

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

#Langchain
from langchain.chains import RetrievalQAWithSourcesChain

#Cross Encoder
from sentence_transformers import CrossEncoder
from typing import List, Tuple



#Streamlit
import streamlit as st
from streamlit_star_rating import st_star_rating


# In[112]:


chroma_client = client = chromadb.PersistentClient(path="C:/Users/Nathan/Kratos_data-Science/Chroma/v7")

# Passing a Chroma Client into Langchain

@st.cache_resource
def load_model():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.success("Embedding model loaded !")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # sementic search
    st.success("Cross Encoder model loaded !")
    return embedding_function, model

embedding_function, model = load_model()




# In[101]:


def create_chroma_instance(collection_name ):
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )


# In[102]:



# In[103]:

def rerank_with_cross_encoder(question: str, retrieved_docs: List[Tuple], top_k: int = 3) -> List[Tuple]:
    """
    Reranks the retrieved documents using the CrossEncoder model.

    :param question: The query question.
    :param retrieved_docs: A list of tuples containing documents and their initial retrieval scores.
    :param top_k: Number of top documents to return after re-ranking.
    :return: Top k documents re-ranked by the CrossEncoder model.
    """
    # Prepare pairs of question and document content for the CrossEncoder
    question_doc_pairs = [(question, doc.page_content) for doc, _ in retrieved_docs]

    # Predict the relevancy scores using the CrossEncoder
    cross_encoder_scores = model.predict(question_doc_pairs)

    # Combine the documents with their new scores
    ranked_docs = [(doc, score) for (doc, _), score in zip(retrieved_docs, cross_encoder_scores)]

    # Sort the documents by their new scores in descending order
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    # Return the top k documents
    return ranked_docs[:top_k]



# Transform the file_path using the replace_and_harmonize_path function
def replace_and_harmonize_path(original_path: str) -> str:
    # Define the old and new beginnings
    old_beginning = "C:/Users/Nathan/Kratos_data-Science/Projects/embeddings"
    new_beginning = "\\\\shackleton\\scratch\\Pour Nathan"
    
    # Replace the old beginning with the new one
    if old_beginning in original_path:
        modified_path = original_path.replace(old_beginning, new_beginning)
    else:
        modified_path = original_path

    # Harmonize the slashes
    harmonized_path = modified_path.replace("/", "\\")
    
    return harmonized_path


# # DOcs retriever

# In[104]:


def reset_page():
    st.session_state.last_question = ""


# In[111]:

# Mapping of display names to their corresponding keys
options_map = {
    "QMS": "QMS-T",
    "Skyminer": "Skyminer-T",
    "EPOCH": "EPOCH-T"
}


def sidebar():
    #Add sidebar on the left
    with st.sidebar:
        st.markdown("# Collection Selection")
        # User can select which collection to query using the modified selectbox
        display_selection = st.selectbox(
            "Choose the collection you want to query:",
            options=list(options_map.keys()),  # Display names
            key='collection_selectbox_unique_key',
        )
        

        # Retrieve the corresponding key for the selected option
        selected_collection = options_map[display_selection]
        #st.markdown(display_selection)

        try:
            langchain_chroma = create_chroma_instance(selected_collection)
            print(f"There are {langchain_chroma._collection.count()} documents in the collection")
        except Exception as e:
            print(f"Failed to create Chroma instance. Error: {e}")
         
        
        x = langchain_chroma._collection.count()  # Assuming you've defined langchain_chroma globally or pass it as an argument.
        num_retrieved_docs = st.sidebar.slider('Select number of retrieved documents:', min_value=1, max_value=100, value=5, key="num_docs_slider")
    
        
        st.markdown("# How to use it")
        st.markdown(
             "1. Select the collection you want to query. \n"
             "2. Enter your question.\n"
             "3. Press Enter.\n"
             "4. Review the document extracts.\n"
             "5. Rate the document extracts.\n"
        )
        st.markdown(
             "For each documents you have :\n"
             "- The document's name\n"
             "- The document's number\n "
             "- An extract of the content\n"
             "- A link to access the document\n"
             "- A score (See 'Score' below)\n"
             "- A rating system to give a note to the document retrieved"
        )
        
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "Cross encoder Retriever is a web application that allows you to retrieve "
            "documents in the database according to your question. The documents can "
            "be PDFs document or HTML pages (like QMS documentation). "
        )
        
        st.markdown("---")
        st.markdown("# Score")
        st.markdown(
            "The score below each document is the cosine similarity\n"
            "score calculated between your question and the content of the documents.\n"

        )
    return selected_collection, num_retrieved_docs, display_selection, langchain_chroma


# In[106]:


# Define your rating function here (assuming you have it in a separate module)
def rating(csv_file, user_question, doc_content, doc_id, collection, score):
    # Attempt to read the existing ratings from the CSV file
    try:
        ratings_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize a new DataFrame with appropriate columns
        ratings_df = pd.DataFrame(columns=['Collection', 'User Question', 'Document Content', 'Document ID', 'Rating', 'Distance', 'Timestamp'])

    # We are generating a unique key based on the document ID for each rating widget to prevent conflicts
    rating_key = f"rating_{doc_id}"
    submit_key = f"submit_{doc_id}"
    
    rating = st_star_rating("Please rate the answer", maxValue=5, defaultValue=0, key=rating_key, size=20) 

    if st.button("Submit", key=submit_key):
        # Add the new rating to the DataFrame
        new_entry = pd.DataFrame({
            'Collection': [collection],
            'User Question': [user_question],
            'Document Content': [doc_content],
            'Document ID': [doc_id],
            'Rating': [rating],
            'Score': [score] ,  
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]  
        })
        ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        ratings_df.to_csv(csv_file, index=False)

        st.write(f"Thank you for the feedback")

    return rating 


# In[107]:


def main():
    st.title("Cross Encoder Retriever")

    # User selects the collection, and the Chroma instance is updated
    selected_collection, num_retrieved_docs, display_selection, langchain_chroma = sidebar()  # Assuming sidebar() is defined elsewhere and works correctly.
    if langchain_chroma is None:
        st.error("Chroma instance is not available. Please check your settings.")
        return
    
    st.write(f"There are {langchain_chroma._collection.count()} documents in the {display_selection} collection")
    
        
    # Ask the user for a question
    user_question = st.text_input("Please enter your question:")

    # Reset the doc_index to the first document when a new question is asked
    if 'last_question' not in st.session_state or st.session_state.last_question != user_question:
        st.session_state.last_question = user_question

    if user_question:  # When the user inputs a question, we process it
        initial_retrieved_docs = langchain_chroma.similarity_search_with_relevance_scores(user_question, k=100) # langchain_chroma._collection.count()
        # Re-rank the documents
        retrieved_docs = rerank_with_cross_encoder(user_question, initial_retrieved_docs, top_k=num_retrieved_docs)
        

        max_docs = len(retrieved_docs)

        # Display the number of retrieved docs
        st.write(f"Retrieved docs: {max_docs}")

        for i, (doc, score) in enumerate(retrieved_docs):
        
            # Extract and display the name of the document from the metadata
            documentation = doc.metadata.get('documentation', 'Not Available')
            source = doc.metadata['source']
            st.write(f"**Documentation:** {documentation} - {source}")

            # Extract the file_path from the metadata
            file_path = doc.metadata.get('file_path', 'Not Available')
            transformed_path = replace_and_harmonize_path(file_path)
            
            st.write(f"**Document {i + 1}/{max_docs}**")

            # Display the document content and details
            st.write(f"**Document Content:** {doc.page_content}")
            
            
            # Display a custom text as a clickable link pointing to the transformed path
            custom_text = "Click here to access the file"
            if "192.168" in transformed_path:
                st.write(f'<a href="{transformed_path}" target="_blank">{custom_text}</a>', unsafe_allow_html=True)
            else:
                st.write(f'<a href="file:{transformed_path}" target="_blank">{custom_text}</a>', unsafe_allow_html=True)
                
                
            st.write(f"**Score:** {round(score, 2)}")

            # Call the rating function for each document
            rating_key = f"rating_doc_{i}"
            submit_key = f"submit_{i}"
            rating_value = rating("C:/Users/Nathan/app/csv_file.csv", user_question, doc.page_content, i, selected_collection, score )
            
            st.markdown("---")
    else:
        st.write("Please enter a question to retrieve relevant documents.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




