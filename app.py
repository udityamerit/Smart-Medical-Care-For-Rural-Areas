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
model, collection, df = load_model_components(DATAFRAME_FILE)


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

    if df is None or model is None or collection is None:
        flash('Sorry, the recommendation service is currently unavailable. Please contact the administrator.', 'danger')
        return render_template('index.html', error="Service unavailable.", query=user_query)

    if not user_query or user_query.isspace():
        if request.args.get('query') == '':
            flash('Please enter a medicine name or a symptom to search.', 'info')
        return render_template('index.html', recommendation=None, error=None, query=None)

    recommended_medicines = get_recommendations(user_query, df, model, collection)
    
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
        previous_search_recommendations = get_previous_search_recommendations(previous_searches, df, model, collection)
    
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
                                query=user_query)
    
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
    
    if df is None or model is None:
         flash('Service unavailable.', 'danger')
         return redirect(url_for('recommender_page'))
         
    # Call the new function for broader/related search
    contextual_meds_df = get_contextual_recommendations(query, df, model, collection)
    
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
    except FileNotFoundError:
        flash('Error: Visualization dataset "final_medicine_dataset_with_age_group.csv" not found in Datasets folder.', 'danger')
        return render_template('dashboard.html', error="Dataset file not found.")
    except Exception as e:
        flash(f'Error loading dataset: {e}', 'danger')
        return render_template('dashboard.html', error=f"Error loading dataset: {e}")

    unique_reasons_set = set()
    for row in df_viz['reason'].dropna():
        reasons = row.split(',')
        for reason in reasons:
            unique_reasons_set.add(reason.strip())
            
    all_reasons = sorted(list(unique_reasons_set))

    selected_reason = 'All'
    selected_plot_type = 'pie' 
    total_medicines = 0
    chart_html = None
    error = None

    if request.method == 'POST':
        selected_reason = request.form.get('reason_filter')
        selected_plot_type = request.form.get('plot_type', 'pie')

    if selected_reason != 'All':
        df_filtered = df_viz[df_viz['reason'].str.contains(selected_reason, case=False, na=False)]
        
        if df_filtered.empty:
            error = f"No data found for the condition: {selected_reason}"
        else:
            total_medicines = len(df_filtered)
            age_breakdown = df_filtered.groupby('age_group').size().reset_index(name='Count')
            chart_title = f'Age Group Breakdown for "{selected_reason}"'

            if selected_plot_type == 'bar':
                fig = px.bar(
                    age_breakdown, x='age_group', y='Count', color='age_group',
                    title=chart_title, template='plotly_dark', text='Count'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_title='Age Group', yaxis_title='Number of Medicines')
            else:
                age_breakdown['legend_label'] = age_breakdown.apply(
                    lambda row: f"{row['age_group'].capitalize()}: {row['Count']}", axis=1
                )
                fig = px.pie(
                    age_breakdown, names='legend_label', values='Count', 
                    title=chart_title, template='plotly_dark'
                )
                fig.update_traces(textposition='inside', textinfo='percent')
            
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
    return render_template(
        'dashboard.html',
        all_reasons=all_reasons,
        selected_reason=selected_reason,
        total_medicines=total_medicines,
        chart_html=chart_html,
        selected_plot_type=selected_plot_type,
        error=error
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
    except Exception as e:
        print(f"Error parsing request: {e}")
        return jsonify({"error": f"Invalid request: {e}"}), 400
    
    pharmacies_with_distance = []
    for pharmacy in pharmacy_data:
        try:
            dist = haversine(user_lat, user_lon, pharmacy['latitude'], pharmacy['longitude'])
            if dist <= SEARCH_RADIUS_KM:
                pharmacies_with_distance.append({
                    'name': pharmacy.get('Name', 'N/A'),
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)