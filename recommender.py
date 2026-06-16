import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def cosine_similarity_1d(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def load_model_components(df_path, collection_name="medicines"):
    """
    Loads the DataFrame and connects to the ChromaDB using LangChain.
    """
    print("--- Initializing Recommender (ChromaDB + LangChain + HuggingFace) ---")
    
    if not os.path.exists(df_path):
        print(f"\nFATAL ERROR: {df_path} not found. Run train_model.py first.")
        return None, None, None

    if not os.path.exists("chroma_db"):
         print("\nFATAL ERROR: 'chroma_db' folder not found. Run train_model.py first.")
         return None, None, None

    try:
        df = pd.read_pickle(df_path)
        
        # Load HuggingFace Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to ChromaDB
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        
        print("Recommender initialized successfully.")
        return embeddings, vector_store, df
        
    except Exception as e:
        print(f"\nERROR loading components: {e}")
        return None, None, None

def generate_ai_explanation(query, recommended_meds):
    """
    Generates a structured medical overview explaining the recommendations using Gemini 1.5 Flash.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        # Format candidate medicines for the prompt
        meds_context = ""
        for i, med in enumerate(recommended_meds):
            meds_context += f"{i+1}. Name: {med['name']}\n   Indications: {med['reason']}\n   Description: {med['description']}\n   Age Group: {med['age_group']}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert AI Clinical Pharmacist. Your task is to analyze the user's health query or medicine search "
                "and explain the recommended medicines. Follow these guidelines:\n"
                "1. Keep your tone empathetic, professional, and clear.\n"
                "2. Provide a brief overview of what the query indicates (e.g. identify if it is a symptom like 'fever' or a drug like 'paracetamol').\n"
                "3. Review the provided candidate medicines and explain which ones are the best matches for the symptoms/query and why.\n"
                "4. List key safety warnings, precautions, and when to seek immediate medical attention.\n"
                "5. Always include a prominent standard medical disclaimer at the very end of your response inside a container styled for a dark themed dashboard. Use exactly this HTML markup:\n"
                "   <div class='medical-disclaimer' style='margin-top: 20px; padding: 15px; border: 1px solid rgba(243, 156, 18, 0.45); background-color: rgba(243, 156, 18, 0.08); color: #fbe5c8; border-radius: 8px; font-weight: 500;'>\n"
                "       <strong>Disclaimer:</strong> This is an AI-generated analysis. Consult a qualified medical practitioner before taking any medication.\n"
                "   </div>\n"
                "Format the rest of the response using clean, semantic HTML tags (e.g., <p>, <strong>, <ul>, <li>) suitable for rendering directly in a web dashboard."
            )),
            ("user", "User Search Query: {query}\n\nCandidate Medicines:\n{meds_context}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query, "meds_context": meds_context})
        return response
    except Exception as e:
        print(f"Error generating AI explanation: {e}")
        return "<p>AI doctor analysis is currently unavailable. Please consult your physician.</p>"

def get_recommendations(query, df, embeddings, vector_store):
    """
    Gets medicine recommendations using LangChain Chroma vector store, ranked via MMR.
    Returns a tuple: (matched_df, ai_explanation)
    """
    try:
        # 1. MMR search on Chroma
        # We fetch 20 candidates and return top 10 diverse matches
        docs = vector_store.max_marginal_relevance_search(query, k=10, fetch_k=20, lambda_mult=0.5)
        
        if not docs:
            return pd.DataFrame(), None

        # 2. Extract database IDs
        db_ids = [int(doc.metadata['db_id']) for doc in docs if 'db_id' in doc.metadata]
        if not db_ids:
            return pd.DataFrame(), None

        matched_df = df.loc[db_ids].copy()

        # 3. Compute exact Cosine Similarity for Match Scores
        query_emb = embeddings.embed_query(query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = embeddings.embed_documents(doc_texts)
        
        scores = []
        for doc_emb in doc_embs:
            score = cosine_similarity_1d(query_emb, doc_emb)
            scores.append(score)

        # 4. Scale scores (85% to 99.6%)
        max_score = max(scores) if scores else 1.0
        scaled_scores = []
        for s in scores:
            relative = max(0.0, float(s)) / max(0.001, max_score)
            boosted = 85.0 + (relative * 13.0) + (min(1.0, float(s)) * 1.5)
            scaled_scores.append(round(min(99.6, boosted)))

        matched_df['Match_Score'] = scaled_scores
        matched_df = matched_df[matched_df['Match_Score'] >= 95]
        
        # 5. Generate AI Explanation for the matched medicines
        recommended_meds = matched_df.head(5).to_dict('records')
        ai_explanation = generate_ai_explanation(query, recommended_meds) if recommended_meds else None

        return matched_df, ai_explanation

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return pd.DataFrame(), None

def get_substitutes(medicine_name, df):
    """
    Gets substitutes for a given medicine.
    """
    substitutes = df[df['name'] == medicine_name][['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']]
    return substitutes.values.flatten().tolist()

def get_contextual_recommendations(query, df, embeddings, vector_store):
    """
    Finds medicines for 'associated conditions' or 'causes' related to the query.
    Uses a broader search query and higher diversity (lower lambda_mult) in MMR.
    """
    try:
        advanced_query = query + " treatments conditions"
        docs = vector_store.max_marginal_relevance_search(advanced_query, k=15, fetch_k=40, lambda_mult=0.6)
        
        if not docs:
            return pd.DataFrame()

        db_ids = [int(doc.metadata['db_id']) for doc in docs if 'db_id' in doc.metadata]
        if not db_ids:
            return pd.DataFrame()

        matched_df = df.loc[db_ids].copy()

        # Compute scores
        query_emb = embeddings.embed_query(advanced_query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = embeddings.embed_documents(doc_texts)
        
        scores = []
        for doc_emb in doc_embs:
            score = cosine_similarity_1d(query_emb, doc_emb)
            scores.append(score)

        max_score = max(scores) if scores else 1.0
        scaled_scores = []
        for s in scores:
            relative = max(0.0, float(s)) / max(0.001, max_score)
            boosted = 85.0 + (relative * 13.0) + (min(1.0, float(s)) * 1.5)
            scaled_scores.append(round(min(99.6, boosted)))

        matched_df['Match_Score'] = scaled_scores
        matched_df = matched_df[matched_df['Match_Score'] >= 95]
        return matched_df

    except Exception as e:
        print(f"Error getting contextual recommendations: {e}")
        return pd.DataFrame()

def get_root_cause_match(query, df, embeddings, vector_store):
    """
    Finds diverse types of a condition or root causes.
    """
    try:
        advanced_query = query + " causes conditions"
        docs = vector_store.max_marginal_relevance_search(advanced_query, k=10, fetch_k=30, lambda_mult=0.7)
        
        if not docs:
            return pd.DataFrame()

        db_ids = [int(doc.metadata['db_id']) for doc in docs if 'db_id' in doc.metadata]
        if not db_ids:
            return pd.DataFrame()

        matched_df = df.loc[db_ids].copy()

        # Compute scores
        query_emb = embeddings.embed_query(advanced_query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = embeddings.embed_documents(doc_texts)
        
        scores = []
        for doc_emb in doc_embs:
            score = cosine_similarity_1d(query_emb, doc_emb)
            scores.append(score)

        max_score = max(scores) if scores else 1.0
        scaled_scores = []
        for s in scores:
            relative = max(0.0, float(s)) / max(0.001, max_score)
            boosted = 85.0 + (relative * 13.0) + (min(1.0, float(s)) * 1.5)
            scaled_scores.append(round(min(99.6, boosted)))

        matched_df['Match_Score'] = scaled_scores
        matched_df = matched_df[matched_df['Match_Score'] >= 95]
        return matched_df
    except Exception as e:
        print(f"Error getting root cause matches: {e}")
        return pd.DataFrame()

def get_previous_search_recommendations(search_history, df, embeddings, vector_store):
    """
    Provides recommendations based on the user's past search queries.
    """
    if not search_history:
        return pd.DataFrame()
        
    try:
        combined_query = " ".join(search_history)
        docs = vector_store.max_marginal_relevance_search(combined_query, k=10, fetch_k=30, lambda_mult=0.5)
        
        if not docs:
            return pd.DataFrame()

        db_ids = [int(doc.metadata['db_id']) for doc in docs if 'db_id' in doc.metadata]
        if not db_ids:
            return pd.DataFrame()

        matched_df = df.loc[db_ids].copy()

        # Compute scores
        query_emb = embeddings.embed_query(combined_query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = embeddings.embed_documents(doc_texts)
        
        scores = []
        for doc_emb in doc_embs:
            score = cosine_similarity_1d(query_emb, doc_emb)
            scores.append(score)

        max_score = max(scores) if scores else 1.0
        scaled_scores = []
        for s in scores:
            relative = max(0.0, float(s)) / max(0.001, max_score)
            boosted = 85.0 + (relative * 13.0) + (min(1.0, float(s)) * 1.5)
            scaled_scores.append(round(min(99.6, boosted)))

        matched_df['Match_Score'] = scaled_scores
        matched_df = matched_df[matched_df['Match_Score'] >= 95]
        return matched_df
    except Exception as e:
        print(f"Error getting previous search recommendations: {e}")
        return pd.DataFrame()