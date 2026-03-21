import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch

def load_model_components(df_path, collection_name="medicines"):
    """
    Loads the DataFrame and connects to the existing ChromaDB.
    """
    # --- GPU CHECK ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Initializing Recommender (ChromaDB + MMR) on {device.upper()} ---")
    
    if not os.path.exists(df_path):
        print(f"\nFATAL ERROR: {df_path} not found. Run train_model.py first.")
        return None, None, None

    if not os.path.exists("chroma_db"):
         print("\nFATAL ERROR: 'chroma_db' folder not found. Run train_model.py first.")
         return None, None, None

    try:
        df = pd.read_pickle(df_path)
        
        # --- GPU MODIFICATION: Load model on GPU ---
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        collection = chroma_client.get_collection(name=collection_name)
        
        print("Recommender initialized successfully.")
        return model, collection, df
        
    except Exception as e:
        print(f"\nERROR loading components: {e}")
        return None, None, None

def mmr_sort(query_embedding, candidate_embeddings, candidate_ids, k=10, lambda_param=0.5):
    """
    Maximal Marginal Relevance (MMR) algorithm.
    """
    if not candidate_ids:
        return []

    query_embedding = np.array(query_embedding).reshape(1, -1)
    candidate_embeddings = np.array(candidate_embeddings)
    
    sim_to_query = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    selected_indices = []
    candidate_indices = list(range(len(candidate_ids)))
    
    while len(selected_indices) < k and candidate_indices:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in candidate_indices:
            relevance = sim_to_query[idx]
            
            if not selected_indices:
                redundancy = 0
            else:
                selected_embeddings = candidate_embeddings[selected_indices]
                current_embedding = candidate_embeddings[idx].reshape(1, -1)
                redundancy = np.max(cosine_similarity(current_embedding, selected_embeddings))
            
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
    return [candidate_ids[i] for i in selected_indices]

def get_recommendations(query, df, model, collection):
    """
    Gets medicine recommendations using Sentence Transformer embeddings 
    stored in ChromaDB, ranked via MMR.
    """
    try:
        # 1. Embed the user query
        query_vec = model.encode([query]).tolist()
        
        # 2. Query ChromaDB
        results = collection.query(
            query_embeddings=query_vec,
            n_results=20,
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        # Safety check: Ensure results are not None or empty
        if results is None:
            return pd.DataFrame()

        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # 3. Apply MMR
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=10, lambda_param=0.5)
        
        # Convert IDs back to integers for DataFrame lookup
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return pd.DataFrame()

def get_substitutes(medicine_name, df):
    """
    Gets substitutes for a given medicine.
    """
    substitutes = df[df['name'] == medicine_name][['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']]
    return substitutes.values.flatten().tolist()

# --- NEW FUNCTIONALITY: Contextual Recommendations ---
def get_contextual_recommendations(query, df, model, collection):
    """
    Finds medicines for 'associated conditions' or 'causes' related to the query.
    Uses a broader search query and higher diversity (lower lambda) in MMR.
    """
    try:
        # Lightly augment query to find deeper associations
        advanced_query = query + " treatments conditions"
        
        query_vec = model.encode([advanced_query]).tolist()
        
        # Fetch more results to explore the 'tail' of the distribution
        results = collection.query(
            query_embeddings=query_vec,
            n_results=50, 
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # Adjust lambda_param (0.6) to ensure relevance while allowing some diversity
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=15, lambda_param=0.6)
        
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]

    except Exception as e:
        print(f"Error getting contextual recommendations: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Test block
    DATAFRAME_FILE = 'processed_data.pkl'
    model, collection, df = load_model_components(DATAFRAME_FILE)
    if df is not None:
        print("Testing Contextual Search for 'Fever'...")
        recs = get_contextual_recommendations("Fever", df, model, collection)
        print(recs[['name', 'reason']].head(5))

# --- NEW FUNCTIONALITY: Root Cause / Diverse Types Match ---
def get_root_cause_match(query, df, model, collection):
    """
    Finds diverse types of a condition (e.g. 'pain' -> stomach pain, body pain, chest pain)
    or root causes.
    """
    try:
        # Lightly augment query to find diverse types and root causes without diluting typos
        advanced_query = query + " causes conditions"
        
        query_vec = model.encode([advanced_query]).tolist()
        
        results = collection.query(
            query_embeddings=query_vec,
            n_results=50, 
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # Use higher lambda_param (0.7) to heavily enforce relevance and prevent wild drift
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=10, lambda_param=0.7)
        
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]
    except Exception as e:
        print(f"Error getting root cause matches: {e}")
        return pd.DataFrame()

# --- NEW FUNCTIONALITY: Previous Search Recommendations ---
def get_previous_search_recommendations(search_history, df, model, collection):
    """
    Provides recommendations based on the user's past search queries.
    """
    if not search_history:
        return pd.DataFrame()
        
    try:
        # Combine previous searches into a single query context
        combined_query = " ".join(search_history)
        
        query_vec = model.encode([combined_query]).tolist()
        
        results = collection.query(
            query_embeddings=query_vec,
            n_results=30, 
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # Standard MMR for general recommendations
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=10, lambda_param=0.5)
        
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]
    except Exception as e:
        print(f"Error getting previous search recommendations: {e}")
        return pd.DataFrame()