import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import torch

def train_and_save_model(data_filepath, df_path, collection_name="medicines"):
    """
    Loads data, generates embeddings using Sentence Transformers (on GPU if available), 
    and stores them in a local ChromaDB collection using batch processing.
    """
    
    # --- GPU CHECK ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Model Training on {device.upper()} ---")

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

    # 2. Save the processed DataFrame
    print(f"Step 2/3: Saving processed dataframe to {df_path}...")
    df.to_pickle(df_path)

    # 3. Generate Embeddings and Store in ChromaDB
    print("Step 3/3: Generating embeddings and storing in ChromaDB...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    # Reset Collection
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception:
        pass # Collection didn't exist
        
    collection = chroma_client.create_collection(name=collection_name)

    # Prepare data lists
    documents = df['soup'].tolist()
    ids = df['id'].tolist()
    metadatas = df[['name', 'reason']].fillna("").to_dict('records')
    
    # Generate embeddings (Batch size 64 for GPU memory safety)
    print(f"Computing embeddings on {device}...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True).tolist()
    
    # --- NEW: Insert into ChromaDB in small batches to avoid InternalError ---
    BATCH_SIZE = 5000
    total_docs = len(documents)
    
    print(f"Inserting {total_docs} records into ChromaDB in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_docs, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, total_docs)
        
        batch_docs = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"  -> Added records {i} to {end_idx}")

    print("\n--- Training Complete! ---")
    print(f"Embeddings stored in 'chroma_db' folder. Collection: {collection_name}")
    print("You can now run 'app.py'.")


if __name__ == '__main__':
    DATA_FILE = "Datasets/final_medicine_dataset_with_age_group.csv"
    DATAFRAME_FILE = 'processed_data.pkl'

    train_and_save_model(DATA_FILE, DATAFRAME_FILE)