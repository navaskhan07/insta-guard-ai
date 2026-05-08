"""
Instagram Fake Account Detection - Train 3 Models
Dataset: train.csv (576), test.csv (120)
Features (11): profile pic, nums/length username, fullname words,
               nums/length fullname, name==username, description length,
               external URL, private, #posts, #followers, #follows
Target: fake (0=Real, 1=Fake)
Models: Random Forest, SVM, Neural Network (TensorFlow - same as notebook)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_all():
    print("=" * 70)
    print("  INSTAGRAM FAKE ACCOUNT DETECTION - MODEL TRAINING")
    print("=" * 70)

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    FEATURES = ['profile pic', 'nums/length username', 'fullname words',
                'nums/length fullname', 'name==username', 'description length',
                'external URL', 'private', '#posts', '#followers', '#follows']

    x_train = train_df[FEATURES]
    y_train = train_df['fake'].values
    x_test = test_df[FEATURES]
    y_test = test_df['fake'].values

    print(f"\n  Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"  Train → Real: {sum(y_train==0)}, Fake: {sum(y_train==1)}")
    print(f"  Test  → Real: {sum(y_test==0)}, Fake: {sum(y_test==1)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(x_train)
    X_test_sc = scaler.transform(x_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    results = {}

    # 1. RANDOM FOREST
    print("\n" + "-" * 60)
    print("  1. RANDOM FOREST")
    print("-" * 60)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf.fit(x_train, y_train)
    rf_train = accuracy_score(y_train, rf.predict(x_train))
    rf_pred = rf.predict(x_test)
    rf_test = accuracy_score(y_test, rf_pred)
    rf_cv = cross_val_score(rf, x_train, y_train, cv=5)
    rf_auc = roc_auc_score(y_test, rf_pred)
    print(f"  Train: {rf_train:.4f} | Test: {rf_test:.4f} | CV: {rf_cv.mean():.4f} | AUC: {rf_auc:.4f}")
    print(f"\n{classification_report(y_test, rf_pred, target_names=['Real','Fake'])}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
    fi = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance:")
    for feat, imp in fi:
        print(f"    {feat:<25s} {imp:.4f} {'█' * int(imp * 40)}")
    joblib.dump(rf, "models/random_forest.pkl")
    results['Random Forest'] = {'train_accuracy': round(rf_train*100,2), 'test_accuracy': round(rf_test*100,2),
                                'cv_score': round(rf_cv.mean()*100,2), 'roc_auc': round(rf_auc*100,2)}

    # 2. SVM
    print("\n" + "-" * 60)
    print("  2. SUPPORT VECTOR MACHINE")
    print("-" * 60)
    clf = GridSearchCV(SVC(probability=True), param_grid={'C': [0.1,1,10,100], 'gamma': [0.001,0.01,0.1,1]},
                       cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
    clf.fit(X_train_sc, y_train)
    svm = clf.best_estimator_
    print(f"  Best: C={svm.C}, gamma={svm.gamma}")
    svm_train = accuracy_score(y_train, svm.predict(X_train_sc))
    svm_pred = svm.predict(X_test_sc)
    svm_test = accuracy_score(y_test, svm_pred)
    svm_cv = cross_val_score(svm, X_train_sc, y_train, cv=5)
    svm_auc = roc_auc_score(y_test, svm_pred)
    print(f"  Train: {svm_train:.4f} | Test: {svm_test:.4f} | CV: {svm_cv.mean():.4f} | AUC: {svm_auc:.4f}")
    print(f"\n{classification_report(y_test, svm_pred, target_names=['Real','Fake'])}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, svm_pred)}")
    joblib.dump(svm, "models/svm.pkl")
    results['SVM'] = {'train_accuracy': round(svm_train*100,2), 'test_accuracy': round(svm_test*100,2),
                      'cv_score': round(svm_cv.mean()*100,2), 'roc_auc': round(svm_auc*100,2)}

    # 3. NEURAL NETWORK (same as notebook)
    print("\n" + "-" * 60)
    print("  3. NEURAL NETWORK (TensorFlow/Keras)")
    print("-" * 60)
    Y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    Y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

    model = Sequential([
        Dense(50, input_dim=11, activation='relu'),
        Dropout(0.3),
        Dense(150, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_sc, Y_train_cat, epochs=50, verbose=1, validation_split=0.1, batch_size=32)

    nn_train = model.evaluate(X_train_sc, Y_train_cat, verbose=0)[1]
    nn_proba = model.predict(X_test_sc, verbose=0)
    nn_pred = np.argmax(nn_proba, axis=1)
    nn_test = accuracy_score(y_test, nn_pred)
    nn_auc = roc_auc_score(y_test, nn_pred)
    print(f"\n  Train: {nn_train:.4f} | Test: {nn_test:.4f} | AUC: {nn_auc:.4f}")
    print(f"\n{classification_report(y_test, nn_pred, target_names=['Real','Fake'])}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, nn_pred)}")
    model.save("models/neural_network.keras")
    results['Neural Network'] = {'train_accuracy': round(nn_train*100,2), 'test_accuracy': round(nn_test*100,2),
                                 'cv_score': round(nn_test*100,2), 'roc_auc': round(nn_auc*100,2)}

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<20} {'Train':>8} {'Test':>8} {'CV':>8} {'AUC':>8}")
    print("  " + "-" * 54)
    for n, r in results.items():
        print(f"  {n:<20} {r['train_accuracy']:>7.2f}% {r['test_accuracy']:>7.2f}% {r['cv_score']:>7.2f}% {r['roc_auc']:>7.2f}%")
    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n  ⭐ Best: {best[0]} ({best[1]['test_accuracy']}%)")

if __name__ == "__main__":
    train_all()
