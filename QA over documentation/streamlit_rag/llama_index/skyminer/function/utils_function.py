import pandas as pd
import streamlit as st
from datetime import datetime
import time
from llama_index import PromptTemplate
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from collections import defaultdict


def feedback(conversation, doc_content, app_name, feedback_dict):
    path = ("C:\\Users\\Bruno.Pinos\\Documents\\virtual-assistant\\QA over "
            "documentation\\streamlit_rag\\llama_index\\ratings.csv")
    # Attempt to read the existing ratings from the CSV file
    try:
        ratings_df = pd.read_csv(path)
    except FileNotFoundError:
        # If the file doesn't exist, initialize a new DataFrame with appropriate columns
        ratings_df = pd.DataFrame(columns=['Application Name', 'Conversation', 'Document Content',
                                           'Rating', 'Timestamp'])
    # Add the new rating to the DataFrame
    new_entry = pd.DataFrame({
        'Application Name': [app_name],
        'Conversation': [conversation],
        'Document Content': [doc_content],
        'Rating': [feedback_dict['score']],
        'Comment': [feedback_dict['text']],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
    # Save the updated DataFrame back to the CSV
    ratings_df.to_csv(path, index=False)
    return 0


def rating(conversation, doc_content, app_name, rating_widget, comment_widget):
    path = ("C:\\Users\\Bruno.Pinos\\Documents\\virtual-assistant\\QA over "
            "documentation\\streamlit_rag\\llama_index\\ratings.csv")
    # Attempt to read the existing ratings from the CSV file
    try:
        ratings_df = pd.read_csv(path)
    except FileNotFoundError:
        # If the file doesn't exist, initialize a new DataFrame with appropriate columns
        ratings_df = pd.DataFrame(columns=['Application Name', 'Conversation', 'Document Content',
                                           'Rating', 'Timestamp'])
    # Add the new rating to the DataFrame
    new_entry = pd.DataFrame({
        'Application Name': [app_name],
        'Conversation': [conversation],
        'Document Content': [doc_content],
        'Rating': [rating_widget],
        'Comment': [comment_widget],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    ratings_df.to_csv(path, index=False)
    st.balloons()
    time.sleep(1)
    return 0


def run_queries(queries, retriever):
    """
    Run queries against retrievers asynchronously.

    :param queries: A list of queries to be processed.
    :param retriever: A retriever that will process the queries.
    :return: A dictionary mapping each query and its index to its corresponding result.
    """
    tasks = []
    for query in queries:
        tasks.append(retriever.retrieve(query))

    results_dict = {}
    # Iterate over each pair of query and its result.
    for i, (query, query_result) in enumerate(zip(queries, tasks)):
        # Map each query and its index to its result in the dictionary.
        results_dict[(query, i)] = query_result

    return results_dict


def fuse_results(results_dict, similarity_top_k: int = 4):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
                sorted(
                    nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
                )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


def generate_queries(llm, query_str: str, query_gen_prompt_str: str, num_queries: int = 3):
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries, query=query_str  # remove the original query
    )
    response = llm.generate([fmt_prompt])

    # Assuming there's only one generation in the response
    if response.generations and len(response.generations[0]) > 0:
        generation_text = response.generations[0][0].text
        queries = generation_text.split("\n")
        return queries
    else:
        return []


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
            self,
            llm,
            retrievers: BaseRetriever,
            query_gen_prompt_str: str,
            similarity_top_k: int = 4,
            num_queries: int = 4
    ) -> None:
        """Init params."""
        self.llm = llm  # Store the llm instance
        self._retrievers = retrievers
        self._query_gen_prompt_str = query_gen_prompt_str
        self._similarity_top_k = similarity_top_k
        self._num_queries = num_queries
        super().__init__()

    def _retrieve(self, query_str):
        queries = generate_queries(self.llm, query_str, self._query_gen_prompt_str, self._num_queries)
        results_dict = run_queries(queries, self._retrievers)
        final_results = fuse_results(results_dict, similarity_top_k=self._similarity_top_k)
        return final_results


class CustomRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        # This is a placeholder. Replace this with your actual retrieval logic.
        nodes_with_scores = self._get_nodes_with_scores(query_bundle)
        # Apply the simple fusion logic
        return self._simple_fusion(nodes_with_scores)

    @staticmethod
    def _simple_fusion(nodes_with_scores):
        all_nodes = {}
        file_name_counts = defaultdict(int)
        for node_with_score in nodes_with_scores:
            file_name = node_with_score.node.metadata["file_name"]
            if file_name_counts[file_name] < 5:
                all_nodes[node_with_score.node.get_content()] = node_with_score
                file_name_counts[file_name] += 1

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)[:20]


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""
    def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
            keyword_retriever: KeywordTableSimpleRetriever,
            mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
