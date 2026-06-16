import os
import time
import random
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Set to None to process all rows, or an integer to process only a subset for testing.
# We set it to None by default so that the entire 248K dataset is indexed during final training.
SUBSET_LIMIT = None  

def train_and_save_model(data_filepath, df_path, collection_name="medicines"):
    """
    Loads data, saves the processed DataFrame, and stores embeddings in ChromaDB using HuggingFace model.
    """
    print("--- Starting Model Training using LangChain & HuggingFace (all-MiniLM-L6-v2) ---")

    # 1. Load and Preprocess Data
    print("Step 1/3: Loading and preprocessing data...")
    try:
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The data file '{data_filepath}' was not found.")
        return

    # Create a 'soup' column
    df['soup'] = df['name'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['reason'].fillna('')

    # Fill NaN values in substitute columns
    sub_cols = ['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']
    for col in sub_cols:
        df[col] = df[col].fillna('')
        
    # Ensure we have a unique ID
    df['id'] = df.index.astype(str)
    
    print(f"Data loaded. {len(df)} records found.")

    # Apply subset limit for development/testing
    if SUBSET_LIMIT is not None:
        print(f"Applying SUBSET_LIMIT of {SUBSET_LIMIT} records for development/testing.")
        df = df.head(SUBSET_LIMIT).copy()
        
    # Save the processed DataFrame (always save to standard path for recommender to load)
    print(f"Step 2/3: Saving processed dataframe to {df_path}...")
    df.to_pickle(df_path)

    # 3. Generate Embeddings and Store in ChromaDB
    print("Step 3/3: Generating embeddings using HuggingFace and storing in ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Chroma Vector Store.
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir):
        print(f"Clearing existing database at '{persist_dir}'...")
        import shutil
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Warning: Could not clear '{persist_dir}': {e}")
            
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Prepare document objects
    print("Preparing document representations...")
    documents = []
    for idx, row in df.iterrows():
        doc_text = row['soup']
        metadata = {
            'name': str(row['name']) if pd.notnull(row['name']) else '',
            'reason': str(row['reason']) if pd.notnull(row['reason']) else '',
            'db_id': str(row['id'])
        }
        documents.append(Document(page_content=doc_text, metadata=metadata, id=str(row['id'])))

    # Insert into ChromaDB in batches
    BATCH_SIZE = 1000  # Local inference is fast, can use larger batch size
    total_docs = len(documents)
    
    print(f"Inserting {total_docs} records into ChromaDB in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_docs, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, total_docs)
        batch_docs = documents[i:end_idx]
        
        print(f"Processing batch {i} to {end_idx}... ", end="", flush=True)
        vector_store.add_documents(documents=batch_docs)
        print("Success.")

    print("\n--- Training Complete! ---")
    print(f"Embeddings stored in '{persist_dir}' folder. Collection: {collection_name}")
    print("You can now run 'app.py'.")

if __name__ == '__main__':
    DATA_FILE = "Datasets/final_medicine_dataset_with_age_group.csv"
    DATAFRAME_FILE = 'processed_data.pkl'

    train_and_save_model(DATA_FILE, DATAFRAME_FILE)