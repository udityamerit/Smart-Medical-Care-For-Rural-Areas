import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import csv
import math
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from recommender import load_model_components, get_recommendations, get_substitutes, get_contextual_recommendations, get_root_cause_match, get_previous_search_recommendations
import pandas as pd
import plotly.express as px
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse

# --- App and Login Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_this'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message_category = 'info'

# --- User Model and Persistent Storage ---
USERS_FILE = 'users.json'

class User(UserMixin):
    def __init__(self, id, username, password, search_history=None):
        self.id = id
        self.username = username
        self.password = password
        self.search_history = search_history if search_history is not None else []

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r') as f:
            users_data = json.load(f)
            return {id: User(id, data['username'], data['password'], data.get('search_history', [])) for id, data in users_data.items()}
    except json.JSONDecodeError:
        print("Error: users.json is corrupted or empty. Creating a new one.")
        return {}
    except Exception as e:
        print(f"Error loading users: {e}")
        return {}


def save_users(users_dict):
    try:
        users_data = {id: {'username': user.username, 'password': user.password, 'search_history': user.search_history} for id, user in users_dict.items()}
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

users = load_users()

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# --- Load Model Components ---
DATAFRAME_FILE = 'processed_data.pkl'
embeddings, vector_store, df = load_model_components(DATAFRAME_FILE)


# --- Haversine Distance Formula ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    rad_lat1 = math.radians(lat1)
    rad_lon1 = math.radians(lon1)
    rad_lat2 = math.radians(lat2)
    rad_lon2 = math.radians(lon2)
    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1
    a = math.sin(d_lat / 2)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

pharmacy_data = []
CSV_FILENAME = 'enriched_pharmacies_corrected.csv'
SEARCH_RADIUS_KM = 15 
try:
    with open(CSV_FILENAME, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['latitude'] = float(row['latitude'])
                row['longitude'] = float(row['longitude'])
                pharmacy_data.append(row)
            except (ValueError, TypeError, KeyError):
                pass
except FileNotFoundError:
    print(f"FATAL ERROR: '{CSV_FILENAME}' not found. Make sure it's in the root folder.")
except Exception as e:
    print(f"FATAL ERROR: Could not load pharmacy data: {e}")


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/medicines-showcase')
def medicines_showcase_page():
    if current_user.is_authenticated:
        return redirect(url_for('medicines_page'))
    
    grouped_medicines = {}
    if df is None or df.empty:
        flash('Sorry, the medicine database is currently unavailable.', 'info')
    else:
        for age_group in df['age_group'].dropna().unique():
            group_df = df[df['age_group'] == age_group]
            sample_size = min(4, len(group_df))
            grouped_medicines[age_group] = group_df.sample(n=sample_size).to_dict('records')
        
    return render_template('medicines_showcase.html', grouped_medicines=grouped_medicines)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('recommender_page'))
    if request.method == 'POST':
        if 'signup_submit' in request.form:
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Username and password are required.', 'danger')
            elif username in [u.username for u in users.values()]:
                flash('Username already exists. Please choose another.', 'danger')
            else:
                new_id = str(len(users) + 1)
                new_user = User(new_id, username, password)
                users[new_id] = new_user
                save_users(users)
                login_user(new_user)
                flash('Account created successfully! You are now logged in.', 'success')
                return redirect(url_for('recommender_page'))
                
        elif 'login_submit' in request.form:
            username = request.form['username_login']
            password = request.form['password_login']
            user = next((u for u in users.values() if u.username == username), None)
            
            if user and user.password == password:
                login_user(user)
                flash('Logged in successfully.', 'success')
                return redirect(url_for('recommender_page'))
            else:
                flash('Invalid username or password.', 'danger')
                
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/recommender', methods=['GET', 'POST'])
@login_required
def recommender_page():
    if request.method == 'POST':
        user_query = request.form.get('query')
        return redirect(url_for('recommender_page', query=user_query))

    user_query = request.args.get('query')

    if df is None or embeddings is None or vector_store is None:
        flash('Sorry, the recommendation service is currently unavailable. Please contact the administrator.', 'danger')
        return render_template('index.html', error="Service unavailable.", query=user_query)

    if not user_query or user_query.isspace():
        if request.args.get('query') == '':
            flash('Please enter a medicine name or a symptom to search.', 'info')
        return render_template('index.html', recommendation=None, error=None, query=None)

    recommended_medicines, ai_explanation = get_recommendations(user_query, df, embeddings, vector_store)
    
    # Manage robust search history to prevent duplicates upon reload
    if not current_user.search_history or current_user.search_history[-1] != user_query:
        if user_query in current_user.search_history:
            current_user.search_history.remove(user_query)
        current_user.search_history.append(user_query)
        if len(current_user.search_history) > 5:
            current_user.search_history.pop(0)
        save_users(users)
        
    previous_search_recommendations = pd.DataFrame()
    # Explicitly pull previous history items (excluding current query)
    previous_searches = [q for q in current_user.search_history if q != user_query]
    if previous_searches:
        previous_search_recommendations = get_previous_search_recommendations(previous_searches, df, embeddings, vector_store)
    
    if not recommended_medicines.empty:
        top_recommendation = recommended_medicines.iloc[0].to_dict()
        substitutes = get_substitutes(top_recommendation['name'], df)
        other_recommendations = recommended_medicines.iloc[1:].to_dict('records') 
        
        return render_template('index.html', 
                                recommendation=top_recommendation, 
                                substitutes=substitutes, 
                                other_recommendations=other_recommendations, 
                                previous_search_recommendations=previous_search_recommendations.to_dict('records') if not previous_search_recommendations.empty else [],
                                previous_searches_list=previous_searches,
                                query=user_query,
                                ai_explanation=ai_explanation)
    
    error_message = f"Sorry, we couldn't find any close matches for '{user_query}'. Please check your spelling or try a different term."
    return render_template('index.html', error=error_message, query=user_query)

# --- NEW ROUTE FOR CONTEXTUAL/ASSOCIATED RECOMMENDATIONS ---
@app.route('/recommended_medicines')
@login_required
def contextual_recommendations():
    query = request.args.get('query')
    if not query:
        flash('No query provided.', 'warning')
        return redirect(url_for('recommender_page'))
    
    if df is None or embeddings is None:
         flash('Service unavailable.', 'danger')
         return redirect(url_for('recommender_page'))
         
    # Call the new function for broader/related search
    contextual_meds_df = get_contextual_recommendations(query, df, embeddings, vector_store)
    
    if contextual_meds_df.empty:
        flash(f'No contextual recommendations found for {query}', 'warning')
        return redirect(url_for('recommender_page'))
        
    contextual_meds = contextual_meds_df.to_dict('records')
    
    return render_template('recommendations.html', 
                           original_query=query, 
                           medicines=contextual_meds)


@app.route('/medicines')
@login_required
def medicines_page():
    grouped_medicines = {}
    if df is None or df.empty:
        flash('Sorry, the medicine database is currently unavailable.', 'danger')
    else:
        for age_group in df['age_group'].dropna().unique():
            group_df = df[df['age_group'] == age_group]
            sample_size = min(8, len(group_df))
            grouped_medicines[age_group] = group_df.sample(n=sample_size).to_dict('records')
        
    return render_template('medicines.html', grouped_medicines=grouped_medicines)

@app.route('/contact')
@login_required
def contact_page():
    return render_template('contact.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard_page():
    file_path = 'Datasets/final_medicine_dataset_with_age_group.csv'
    
    try:
        df_viz = pd.read_csv(file_path)
    except Exception as e:
        flash(f'Error loading dataset: {e}', 'danger')
        return render_template('dashboard.html', error=f"Error loading dataset: {e}")

    # Extract all unique reasons (conditions)
    unique_reasons_set = set()
    for row in df_viz['reason'].dropna():
        reasons = [r.strip() for r in str(row).split(',')]
        unique_reasons_set.update(reasons)
    all_reasons = sorted(list(unique_reasons_set))

    selected_reason = request.form.get('reason_filter', 'All')
    
    # 1. Prepare Global Analytics (Top medicines by versatility - count of reasons)
    # This shows "Top Quality" graph data for users to see which meds are most versatile
    df_viz['reason_count'] = df_viz['reason'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    top_versatile = df_viz.nlargest(10, 'reason_count')[['name', 'reason_count']].to_dict('records')
    
    # 2. Filter data if a reason is selected
    age_dist = []
    condition_stats = {"total": len(df_viz), "selected_count": 0}
    top_meds_for_condition = []

    # 2. Filter data if a reason is selected
    age_dist = []
    condition_stats = {
        "total": len(df_viz), 
        "selected_count": 0,
        "avg_substitutes": 0,
        "primary_system": "N/A"
    }
    top_meds_for_condition = []
    system_impact = {
        "Respiratory": 0,
        "Nervous System": 0,
        "Cardiovascular": 0,
        "Gastrointestinal": 0,
        "Infection/Immune": 0,
        "Musculoskeletal": 0
    }

    if selected_reason != 'All':
        df_filtered = df_viz[df_viz['reason'].str.contains(selected_reason, case=False, na=False)].copy()
        condition_stats["selected_count"] = len(df_filtered)
        
        if not df_filtered.empty:
            # Age Group Distribution
            age_dist = df_filtered.groupby('age_group').size().reset_index(name='count').to_dict('records')
            
            # Top Medicines for this specific condition
            top_meds_for_condition = df_filtered.head(8)[['name', 'age_group']].to_dict('records')

            # --- ADVANCED ANALYSIS: Substitution Density ---
            sub_cols = ['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']
            df_filtered['sub_count'] = df_filtered[sub_cols].notnull().sum(axis=1)
            condition_stats["avg_substitutes"] = round(df_filtered['sub_count'].mean(), 1)

            # --- ADVANCED ANALYSIS: Biological System Impact ---
            system_keywords = {
                "Respiratory": ["lung", "breath", "respiratory", "cough", "nose", "throat", "airways", "asthma", "sinus"],
                "Nervous System": ["anxiety", "brain", "nerve", "neuropathic", "central nervous", "sleep", "insomnia", "panic", "depression", "seizure"],
                "Cardiovascular": ["heart", "blood pressure", "hypertension", "cholesterol", "circulatory", "cardiac", "artery", "stroke"],
                "Gastrointestinal": ["stomach", "gut", "acid reflux", "digestion", "bowel", "ulcer", "piles", "nausea", "vomiting", "diarrhea"],
                "Infection/Immune": ["bacterial", "fungal", "parasitic", "infection", "virus", "immune", "antibiotic", "microbial"],
                "Musculoskeletal": ["pain relief", "muscle", "bone", "joint", "inflammation", "arthritis", "fever", "ache"]
            }

            for desc in df_filtered['description'].dropna():
                desc_lower = str(desc).lower()
                for system, keywords in system_keywords.items():
                    if any(kw in desc_lower for kw in keywords):
                        system_impact[system] += 1
            
            # Find primary system
            if any(system_impact.values()):
                condition_stats["primary_system"] = max(system_impact, key=system_impact.get)

    return render_template(
        'dashboard.html',
        all_reasons=all_reasons,
        selected_reason=selected_reason,
        top_versatile=top_versatile,
        age_dist=age_dist,
        condition_stats=condition_stats,
        top_meds_for_condition=top_meds_for_condition,
        system_impact=system_impact
    )

@app.route('/pharmacy-finder')
@login_required
def pharmacy_finder_page():
    return render_template('pharmacy_finder.html')


@app.route('/api/find-pharmacies', methods=['POST'])
def find_pharmacies():
    try:
        data = request.json
        user_lat = float(data.get('user_lat'))
        user_lon = float(data.get('user_lon'))
        # Get radius from request, fallback to global SEARCH_RADIUS_KM
        radius_km = float(data.get('radius_km', SEARCH_RADIUS_KM))
    except Exception as e:
        print(f"Error parsing request: {e}")
        return jsonify({"error": f"Invalid request: {e}"}), 400
    
    pharmacies_with_distance = []
    for pharmacy in pharmacy_data:
        try:
            dist = haversine(user_lat, user_lon, pharmacy['latitude'], pharmacy['longitude'])
            if dist <= radius_km:
                pharmacies_with_distance.append({
                    'name': pharmacy.get('Name').strip() if pharmacy.get('Name') and pharmacy.get('Name').strip() else 'Local Pharmacy',
                    'address': pharmacy.get('Address', ''),
                    'distance_km': dist,
                    'google_map_link': pharmacy.get('Google_Maps_Link', '#'),
                    'latitude': pharmacy['latitude'],
                    'longitude': pharmacy['longitude']
                })
        except Exception:
            pass
            
    if not pharmacies_with_distance:
        return jsonify([])

    sorted_pharmacies = sorted(pharmacies_with_distance, key=lambda p: p['distance_km'])
    return jsonify(sorted_pharmacies)

@app.route('/api/medicine-info', methods=['POST'])
def get_medicine_info():
    data = request.json
    medicine_name = data.get('medicine_name', '').strip()
    
    if not medicine_name:
        return jsonify({"error": "Medicine name is required"}), 400
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert clinical pharmacist. Provide important safety warnings, side effects, "
                "and precautions for the medicine requested. Follow these constraints:\n"
                "1. Strictly only discuss safety warnings, side effects, precautions, and warning signs for this specific medicine.\n"
                "2. Provide 3-5 clear, bulleted points.\n"
                "3. Highlight key risk terms like 'side effects', 'warnings', 'allergic reactions', 'cautions', 'avoid', 'severe', 'fatal' "
                "using standard bold tags (`<strong>`).\n"
                "4. Always include a disclaimer at the end to consult a medical practitioner.\n"
                "Format the response using clean, simple HTML tags (e.g., <p>, <strong>, <ul>, <li>) so it renders nicely in a modal."
            )),
            ("user", "Medicine Name: {medicine_name}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        info_html = chain.invoke({"medicine_name": medicine_name})
        
        return jsonify({"info": info_html})
        
    except Exception as e:
        print(f"Error fetching medicine info: {e}")
        return jsonify({"error": "Failed to retrieve safety information. Please try again later."}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)