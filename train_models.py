import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("==> Starting model training...")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ['profile pic', 'nums/length username', 'fullname words',
            'nums/length fullname', 'name==username', 'description length',
            'external URL', 'private', '#posts', '#followers', '#follows']

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

X_train = train_df[FEATURES].values
y_train = train_df['fake'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['fake'].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print("==> Scaler saved.")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
print(f"==> Random Forest trained. Accuracy: {rf_acc:.2f}%")

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_sc)) * 100
joblib.dump(svm, os.path.join(MODELS_DIR, 'svm.pkl'))
print(f"==> SVM trained. Accuracy: {svm_acc:.2f}%")

nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn.fit(X_train_sc, y_train)
nn_acc = accuracy_score(y_test, nn.predict(X_test_sc)) * 100
joblib.dump(nn, os.path.join(MODELS_DIR, 'neural_network.pkl'))
print(f"==> Neural Network trained. Accuracy: {nn_acc:.2f}%")

results = {
    "Random Forest": {"accuracy": round(rf_acc, 2)},
    "SVM":           {"accuracy": round(svm_acc, 2)},
    "Neural Network":{"accuracy": round(nn_acc, 2)}
}
with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
    json.dump(results, f)

print("==> All models trained and saved successfully!")