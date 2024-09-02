import pandas as pd
from datetime import datetime


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