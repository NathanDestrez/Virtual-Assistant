import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
import nltk
from streamlit_feedback import streamlit_feedback
import re


def replace_big_space_with_double_space(input_string):
    # Use regular expression to replace multiple consecutive spaces with a double space
    output_string = re.sub(r' {2,}', '  \n\n', input_string)
    return output_string


def _submit_feedback(user_feedback, emoji=None):
    st.toast(f"Feedback submitted: {user_feedback}", icon=emoji)
    return user_feedback


def display_message(messages, k, app_name='virtual_assistant', remove_space=True):
    for n, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if remove_space:
                message_content = replace_big_space_with_double_space(message["content"])
            else:
                message_content = message["content"]
            st.write(message_content, unsafe_allow_html=True)
            if "sources" in message.keys():
                st.markdown(message["sources"], unsafe_allow_html=True)
        if message["role"] == "assistant" and n > 0:
            feedback_key = f"feedback_{n + k}"
            user_response = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="Please provide extra information",
                on_submit=_submit_feedback,
                key=feedback_key,
            )
            if user_response:
                if user_response['score'] == '👎':
                    user_response['score'] = 0
                else:
                    user_response['score'] = 1
                feedback(messages, app_name, user_response)


def add_sources(context_, response, score=False):
    """
    Adds sources information to the response and displays it using Markdown.

    Parameters:
    - context_ (list): List of nodes containing sources.
    - response (str): The response content to be displayed.

    Returns:
    dict: A dictionary containing information about the response and sources.

    Example:
    context = [...]  # List of nodes with metadata containing source information
    response_text = "The assistant's response."
    result = add_sources(context, response_text)
    print(result)
    {"role": "assistant", "content": "The assistant's response.", "sources": "The sources with link and page if pdf"}
    """
    if len(context_) > 0:
        sources_markdown = "##### Sources:\n\n"
        list_url = list()
        for i, node in enumerate(context_):
            if 'url' in node.metadata.keys():
                url = node.metadata['url']
                if url in list_url:
                    continue
                list_url.append(url)
                if 'page_label' in node.metadata.keys():
                    source_link = (f"<a href='{url}' target='_blank'>{url.split('/')[-1]}</a>"
                                   f" / page: {node.metadata['page_label']}")
                else:
                    source_link = f"<a href='{url}' target='_blank'>{url.split('/')[-1]}</a>"
            elif 'file_path' in node.metadata.keys():
                source_link = f"Path: **{node.metadata['file_path']}**"
            else:
                continue
            if score:
                source_link += f" / Score: {round(node.score, 2)}"
            sources_markdown += f"{i + 1}. {source_link}<br>\n"
        st.markdown(sources_markdown, unsafe_allow_html=True)
        message = {"role": "assistant", "content": response, "sources": sources_markdown}
    else:
        message = {"role": "assistant", "content": response}
    return message


def context_retriever(index, llm, prompt, context_prompt, max_sources=6, score_threshold=0.3):
    """
    Retrieves relevant documents from an index based on a query generated from language models.
    Filters and sorts the retrieved documents based on their scores and relevance.

    Parameters: - index (DocumentIndex): An index containing documents for retrieval. - llm (LanguageModel): A
    language model used to generate queries from prompts. - prompt (str): The primary prompt used for generating the
    query. - context_prompt (str): Additional prompt for context information. - max_sources (int, optional): Maximum
    number of sources/documents to retrieve. Defaults to 6. - score_threshold (float, optional): Minimum score
    required for a document to be considered relevant. Defaults to 0.3.

    Returns:
    tuple: A tuple containing two elements:
        1. A formatted string containing metadata and text information of the selected documents.
        2. A list of Document objects representing the selected documents.
    """
    query = llm(context_prompt)
    query = query.split('(')[0].split('?')[0]
    # Document selection
    retriever = index.as_retriever(similarity_top_k=np.ceil(max_sources/2))
    context_list = retriever.retrieve(query)
    context_list += retriever.retrieve(prompt)
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
    score_list = list(score_list)
    context_ = list(context_)
    # Max 3 documents cause of 2048 token mistral limit
    if len(context_) > max_sources:
        context_ = context_[:max_sources]
        score_list = score_list[:max_sources]
    # Remove bad documents
    indices_to_remove = [i for i, score in enumerate(score_list) if score < score_threshold]
    for index in reversed(indices_to_remove):
        del score_list[index]
        del context_[index]
    context = ""
    for n, doc in enumerate(context_):
        metadata_keys = list(set(doc.metadata) - set(doc.node.excluded_llm_metadata_keys))
        context += f"- "
        for key in metadata_keys:
            context += f'{key}: {doc.metadata[key]} \n '
        context += 'text: ' + doc.text + ' ] \n\n '
    return context, context_


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def count_tokens(text):
    """
    This function uses Natural Language Toolkit (NLTK) to tokenize a given text,
    then counts the number of tokens.
    :param text: str - The input string for which we want to count the number of tokens.
    return int - Returns an integer representing the total number of tokens in the provided text.
    """
    tokens = nltk.word_tokenize(text)
    num_tokens = len(tokens)
    return num_tokens


def feedback(conversation, app_name, feedback_dict):
    path = "ratings.csv"
    # Attempt to read the existing ratings from the CSV file
    try:
        ratings_df = pd.read_csv(path)
    except FileNotFoundError:
        # If the file doesn't exist, initialize a new DataFrame with appropriate columns
        ratings_df = pd.DataFrame(columns=['Application Name', 'Conversation', 'Rating', 'Timestamp'])
    # Add the new rating to the DataFrame
    new_entry = pd.DataFrame({
        'Application Name': [app_name],
        'Conversation': [conversation],
        'Rating': [feedback_dict['score']],
        'Comment': [feedback_dict['text']],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
    # Save the updated DataFrame back to the CSV
    ratings_df.to_csv(path, index=False)
    return 0
