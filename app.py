import os
import hashlib
import secrets
import shutil
import threading
import time
import uuid
import tempfile
from datetime import timedelta
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Load .env for local development (no-op in production if vars already set)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # override=False: env vars already set take priority
    print('[AIOPharmacy] Loaded .env file')
except ImportError:
    pass

import csv
import math
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
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
from werkzeug.middleware.proxy_fix import ProxyFix

# --- App and Login Configuration ---
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# ---------------------------------------------------------------
# SECRET KEY — stable across restarts so existing cookies stay valid.
#
# PRIORITY ORDER:
#   1. SECRET_KEY env variable (set as HF Space Secret — most secure)
#   2. Stable HMAC key derived from SPACE_ID (survives restarts on same Space)
#   3. Random per-session key (local dev only — users log out on restart)
#
# WHY THIS MATTERS: If the key changes every restart, all session cookies
# become invalid → every user is instantly logged out on every reboot.
# ---------------------------------------------------------------
_space_id = os.environ.get('SPACE_ID', '')
if os.environ.get('SECRET_KEY'):
    # Best: explicit secret set by admin
    app.secret_key = os.environ['SECRET_KEY']
elif _space_id:
    # Good: deterministic key derived from Space ID — survives container restarts
    app.secret_key = hashlib.sha256(
        f"aiopharmacy-stable-secret-v2-{_space_id}".encode()
    ).hexdigest()
else:
    # Fallback for local dev: random per-session (users log out on restart, acceptable locally)
    app.secret_key = secrets.token_hex(32)

# ---------------------------------------------------------------
# SESSION ISOLATION — every browser gets its own signed cookie.
# These flags prevent one user's session leaking to another.
# ---------------------------------------------------------------
app.config['SESSION_COOKIE_HTTPONLY'] = True       # JS cannot read the cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'      # blocks cross-site cookie sending
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)   # sessions last 30 days
app.config['SESSION_COOKIE_NAME'] = 'aiopharmacy_session'       # unique cookie name
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Remember-me cookie: persists across browser closes for 30 days
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_NAME'] = 'aiopharmacy_remember'

CORS(app)

# Hugging Face Spaces runs behind HTTPS — enable Secure flag so
# cookies are only sent over encrypted connections.
if _space_id:
    app.config['SESSION_COOKIE_SAMESITE'] = 'None'   # needed for iframe embed on HF
    app.config['SESSION_COOKIE_SECURE'] = True        # HTTPS only
    app.config['REMEMBER_COOKIE_SECURE'] = True

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message = 'Please log in to access AIOPharmacy.'
login_manager.login_message_category = 'info'

# ---------------------------------------------------------------
# PERSISTENT USER STORAGE  
#
# Storage architecture (in priority order):
#
# 1. HF DATASET REPO (primary, fully persistent)
#    Push/pull to a DATASET repo (repo_type='dataset').
#    CRITICAL: Dataset pushes do NOT trigger Space rebuilds!
#    This is the only safe way to persist data across HF restarts.
#    Requires: HF_TOKEN + DATASET_ID env vars.
#
# 2. LOCAL FILE CACHE (secondary, fast reads)
#    /data on HF paid tier, or ./ locally.
#    Acts as a write-through cache — fast reads, synced to dataset on write.
#
# WHY NOT SPACE GIT REPO:
#    Pushing to the Space repo triggers a full container rebuild + restart.
#    This was the original bug causing everyone to be logged out.
# ---------------------------------------------------------------
_HF_TOKEN  = os.environ.get('HF_TOKEN', '')
_DATASET_ID = os.environ.get('DATASET_ID', '')   # e.g. "udityanarayan/aiopharmacy-data"
_DATA_DIR  = '/data' if os.path.isdir('/data') else '.'
USERS_FILE = os.path.join(_DATA_DIR, 'users.json')

# Debounce: track last dataset push to avoid flooding API on search-history writes
_last_dataset_sync: float = 0.0
_DATASET_SYNC_MIN_INTERVAL = 30.0   # seconds between dataset pushes

def _sync_users_from_dataset() -> None:
    """Download users.json from HF Dataset on startup. Thread-safe startup call."""
    if not (_HF_TOKEN and _DATASET_ID):
        print('[AIOPharmacy] No DATASET_ID/HF_TOKEN — skipping dataset sync (local mode)')
        return
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=_DATASET_ID,
            filename='users.json',
            repo_type='dataset',   # dataset push — NO Space rebuild triggered
            token=_HF_TOKEN,
            force_download=True    # always pull latest on startup
        )
        shutil.copy(downloaded, USERS_FILE)
        print(f'[AIOPharmacy] Synced users.json from dataset {_DATASET_ID}')
    except Exception as e:
        print(f'[AIOPharmacy] Dataset sync on startup failed (using local): {e}')

def _sync_users_to_dataset(important: bool = False) -> None:
    """
    Push users.json to HF Dataset (does NOT trigger Space rebuilds).
    For unimportant writes (e.g. search history), debounces to once per 30s.
    For important writes (e.g. new account, password change), always syncs immediately.
    """
    global _last_dataset_sync
    if not (_HF_TOKEN and _DATASET_ID and os.path.exists(USERS_FILE)):
        return
    now = time.time()
    if not important and (now - _last_dataset_sync) < _DATASET_SYNC_MIN_INTERVAL:
        return   # debounced — skip this write
    _last_dataset_sync = now
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=USERS_FILE,
            path_in_repo='users.json',
            repo_id=_DATASET_ID,
            repo_type='dataset',   # CRITICAL: dataset, not space — no rebuild!
            token=_HF_TOKEN,
            commit_message='AIOPharmacy: user DB update'
        )
        print(f'[AIOPharmacy] Synced users.json to dataset {_DATASET_ID}')
    except Exception as e:
        print(f'[AIOPharmacy] Warning: could not sync users to dataset: {e}')

# On first boot, if local cache is empty, seed from bundled users.json
_BUNDLED_USERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')
if not os.path.exists(USERS_FILE) and os.path.exists(_BUNDLED_USERS) and _DATA_DIR != '.':
    try:
        shutil.copy(_BUNDLED_USERS, USERS_FILE)
        print(f'[AIOPharmacy] Seeded {USERS_FILE} from bundled users.json')
    except Exception as _e:
        print(f'[AIOPharmacy] Could not seed users.json: {_e}')

# Pull latest user DB from dataset (important: runs before any request)
_sync_users_from_dataset()
print(f'[AIOPharmacy] User database path: {USERS_FILE}')

class User(UserMixin):
    def __init__(self, id, username, password_hash, search_history=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.search_history = search_history if search_history is not None else []

# ---------------------------------------------------------------
# THREAD-SAFE USER STORAGE
# A single RLock serialises ALL reads and writes to users.json.
# This prevents race conditions when multiple users act at once.
# ---------------------------------------------------------------
_users_lock = threading.RLock()

def _is_plain_text_password(value: str) -> bool:
    """Detect plain-text password (not a werkzeug hash)."""
    return not value.startswith(('pbkdf2:', 'scrypt:', 'bcrypt:', '$2b$'))

def _load_users_unsafe() -> dict:
    """Load users from disk. MUST be called while holding _users_lock."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        users_dict = {}
        migrated = False

        # --- Migration: integer keys (old format) → UUID keys ---
        # Old accounts had sequential integer keys like "1", "2".
        # These must be upgraded to UUIDs so Flask-Login can reliably
        # look them up across worker processes.
        old_int_keys = [k for k in data if k.isdigit()]
        for old_key in old_int_keys:
            new_key = str(uuid.uuid4())
            data[new_key] = data.pop(old_key)
            migrated = True
            print(f'Migrated user key {old_key!r} → {new_key!r}')

        for uid, d in data.items():
            stored = d.get('password', '')
            if _is_plain_text_password(stored):
                stored = generate_password_hash(stored)
                data[uid]['password'] = stored
                migrated = True
            users_dict[uid] = User(uid, d['username'], stored, d.get('search_history', []))
        if migrated:
            _write_users_unsafe(users_dict)
            print('User database migration complete (keys/passwords updated).')
        return users_dict
    except (json.JSONDecodeError, Exception) as e:
        print(f'Error loading users: {e}')
        return {}

def _write_users_unsafe(users_dict: dict) -> None:
    """
    Atomic write: serialise to a temp file then os.replace() so a crash
    mid-write can NEVER leave users.json in a corrupted state.
    Writes to the persistent /data directory on HF Spaces.
    MUST be called while holding _users_lock.
    """
    payload = {
        uid: {
            'username': u.username,
            'password': u.password_hash,
            'search_history': u.search_history
        }
        for uid, u in users_dict.items()
    }
    dir_ = os.path.dirname(os.path.abspath(USERS_FILE)) or '.'
    # Ensure directory exists (important for /data on first boot)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4)
        os.replace(tmp_path, USERS_FILE)   # atomic on POSIX and Windows
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    # Persist to HF Dataset in a non-blocking background thread.
    # important=True so new writes (registration, password change) sync immediately.
    threading.Thread(
        target=_sync_users_to_dataset,
        kwargs={'important': True},
        daemon=True
    ).start()

# ---- Public thread-safe helpers (use these everywhere in routes) ----

def get_user_by_id(user_id: str):
    """Return a User or None. Safe for concurrent calls."""
    with _users_lock:
        return _load_users_unsafe().get(user_id)

def get_user_by_username(username: str):
    """Return a User or None. Safe for concurrent calls."""
    with _users_lock:
        users = _load_users_unsafe()
        return next((u for u in users.values() if u.username == username), None)

def create_user(username: str, password: str):
    """
    Atomically create a new user.
    Returns (User, None) on success or (None, error_message) on failure.
    UUID-based IDs eliminate the len()+1 race condition.
    """
    with _users_lock:
        users = _load_users_unsafe()
        if any(u.username == username for u in users.values()):
            return None, 'Username already exists. Please choose another.'
        new_id = str(uuid.uuid4())
        new_user = User(new_id, username, generate_password_hash(password))
        users[new_id] = new_user
        _write_users_unsafe(users)
        return new_user, None

def update_user_password(user_id: str, new_password: str) -> bool:
    """Hash and save a new password for a single user. Thread-safe."""
    with _users_lock:
        users = _load_users_unsafe()
        if user_id not in users:
            return False
        users[user_id].password_hash = generate_password_hash(new_password)
        _write_users_unsafe(users)
        return True

def update_user_history(user_id: str, history: list) -> bool:
    """
    Persist updated search history for a single user. Thread-safe.
    Uses a debounced (non-important) dataset sync so search queries
    don't flood the HF Dataset API — syncs at most once per 30 seconds.
    """
    with _users_lock:
        users = _load_users_unsafe()
        if user_id not in users:
            return False
        users[user_id].search_history = history
        # Write locally (fast) but use debounced dataset sync
        payload = {
            uid: {
                'username': u.username,
                'password': u.password_hash,
                'search_history': u.search_history
            }
            for uid, u in users.items()
        }
        dir_ = os.path.dirname(os.path.abspath(USERS_FILE)) or '.'
        os.makedirs(dir_, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4)
            os.replace(tmp_path, USERS_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        # Debounced background sync — not critical, history loss on restart is acceptable
        threading.Thread(
            target=_sync_users_to_dataset,
            kwargs={'important': False},
            daemon=True
        ).start()
        return True

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login callback — called on every authenticated request."""
    return get_user_by_id(user_id)

@app.before_request
def make_session_permanent():
    """
    Mark every session as permanent so Flask respects PERMANENT_SESSION_LIFETIME (30 days).
    Without this, PERMANENT_SESSION_LIFETIME is ignored and sessions expire
    when the browser tab closes — causing apparent 'logouts' at any browser close.
    """
    from flask import session
    session.permanent = True

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

    active_form = None
    if request.method == 'POST':
        if 'signup_submit' in request.form:
            active_form = 'signup'
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')

            if not username or not password:
                flash('Username and password are required.', 'danger')
            else:
                new_user, err = create_user(username, password)
                if err:
                    flash(err, 'danger')
                else:
                    # remember=True: sets a persistent cookie lasting 30 days
                    # so the user stays logged in across browser restarts and
                    # container reboots (as long as SECRET_KEY stays stable).
                    login_user(new_user, remember=True)
                    flash('Account created successfully! Welcome to AIOPharmacy 🎉', 'success')
                    return redirect(url_for('recommender_page'))

        elif 'login_submit' in request.form:
            active_form = 'login'
            username = request.form.get('username_login', '').strip()
            password = request.form.get('password_login', '')
            user = get_user_by_username(username)

            if user and check_password_hash(user.password_hash, password):
                login_user(user, remember=True)  # persistent 30-day cookie
                flash('Welcome back! You are now logged in.', 'success')
                return redirect(url_for('recommender_page'))
            else:
                flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html', active_form=active_form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_page():
    """
    Serves only the currently logged-in user's own data — fully thread-safe.
    """
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'change_password':
            current_password = request.form.get('current_password', '')
            new_password     = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')

            me = get_user_by_id(current_user.id)
            if not me:
                flash('Session error. Please log in again.', 'danger')
                return redirect(url_for('login_page'))

            if not check_password_hash(me.password_hash, current_password):
                flash('Current password is incorrect.', 'danger')
            elif len(new_password) < 4:
                flash('New password must be at least 4 characters.', 'danger')
            elif new_password != confirm_password:
                flash('New passwords do not match.', 'danger')
            else:
                update_user_password(current_user.id, new_password)
                flash('Password changed successfully!', 'success')

        elif action == 'clear_history':
            update_user_history(current_user.id, [])
            flash('Search history cleared.', 'success')

        return redirect(url_for('profile_page'))

    me = get_user_by_id(current_user.id)
    return render_template('profile.html',
                           username=me.username if me else current_user.username,
                           search_history=me.search_history if me else [])

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
    
    # Manage robust search history — read, mutate, and atomically write back to disk
    me = get_user_by_id(current_user.id)
    if me:
        history = list(me.search_history)  # fresh copy from disk
        if not history or history[-1] != user_query:
            if user_query in history:
                history.remove(user_query)
            history.append(user_query)
            if len(history) > 5:
                history.pop(0)
            update_user_history(current_user.id, history)
            me.search_history = history  # keep local reference in sync
        
    previous_search_recommendations = pd.DataFrame()
    # Explicitly pull previous history items (excluding current query) — use fresh disk data
    fresh_history = me.search_history if me else []
    previous_searches = [q for q in fresh_history if q != user_query]
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
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)