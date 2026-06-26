import os, json
import numpy as np
import joblib
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'instaGuard-secret-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:////tmp/fakeguard.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    fullname   = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PredictionLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_used = db.Column(db.String(50))
    input_data = db.Column(db.Text)
    result     = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
rf_model  = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm.pkl'))
nn_model  = joblib.load(os.path.join(MODELS_DIR, 'neural_network.pkl'))
scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

with open(os.path.join(MODELS_DIR, 'results.json')) as f:
    model_results = json.load(f)

@app.template_filter('from_json')
def from_json_filter(s):
    try: return json.loads(s)
    except: return {}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        if not fullname or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        user = User(fullname=fullname, email=email,
                    password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id']   = user.id
            session['user_name'] = user.fullname
            flash(f'Welcome back, {user.fullname}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user       = User.query.get(session['user_id'])
    logs       = PredictionLog.query.filter_by(user_id=user.id)\
                     .order_by(PredictionLog.created_at.desc()).limit(10).all()
    total      = PredictionLog.query.filter_by(user_id=user.id).count()
    fake_count = PredictionLog.query.filter_by(user_id=user.id, result='FAKE').count()
    real_count = PredictionLog.query.filter_by(user_id=user.id, result='REAL').count()
    return render_template('dashboard.html', user=user, model_results=model_results,
                           logs=logs, total_predictions=total,
                           fake_count=fake_count, real_count=real_count)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            features = np.array([[
                int(request.form.get('profile_pic', 0)),
                float(request.form.get('nums_length_username', 0)),
                int(request.form.get('fullname_words', 0)),
                float(request.form.get('nums_length_fullname', 0)),
                int(request.form.get('name_eq_username', 0)),
                int(request.form.get('description_length', 0)),
                int(request.form.get('external_url', 0)),
                int(request.form.get('private', 0)),
                int(request.form.get('posts', 0)),
                int(request.form.get('followers', 0)),
                int(request.form.get('follows', 0)),
            ]])

            model_choice = request.form.get('model', 'random_forest')

            if model_choice == 'random_forest':
                prediction = rf_model.predict(features)[0]
                proba      = rf_model.predict_proba(features)[0]
                confidence = round(max(proba) * 100, 2)
                model_name = 'Random Forest'
            elif model_choice == 'svm':
                features_sc = scaler.transform(features)
                prediction  = svm_model.predict(features_sc)[0]
                proba       = svm_model.predict_proba(features_sc)[0]
                confidence  = round(max(proba) * 100, 2)
                model_name  = 'SVM'
            else:
                features_sc = scaler.transform(features)
                prediction  = nn_model.predict(features_sc)[0]
                proba       = nn_model.predict_proba(features_sc)[0]
                confidence  = round(max(proba) * 100, 2)
                model_name  = 'Neural Network'

            label  = "FAKE" if prediction == 1 else "REAL"
            result = {'label': label, 'confidence': confidence,
                      'model_name': model_name, 'is_fake': prediction == 1}

            log = PredictionLog(user_id=session['user_id'], model_used=model_name,
                                input_data=json.dumps({}), result=label, confidence=confidence)
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
    return render_template('predict.html', result=result)

@app.route('/history')
@login_required
def history():
    logs = PredictionLog.query.filter_by(user_id=session['user_id'])\
               .order_by(PredictionLog.created_at.desc()).all()
    return render_template('history.html', logs=logs)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)