# InstaGuard AI - Instagram Fake Account Detector

## Overview
ML-powered web application to detect fake Instagram accounts using 3 trained models:
- **Random Forest** — 91.67% test accuracy ⭐ (Best)
- **SVM** (Support Vector Machine) — 87.50% test accuracy
- **Neural Network** (TensorFlow/Keras) — 87.50% test accuracy

## Dataset
- **Train**: 576 Instagram profiles (288 Real + 288 Fake)
- **Test**: 120 Instagram profiles (60 Real + 60 Fake)

### 11 Features
| Feature | Description | Type |
|---------|-------------|------|
| profile pic | Has profile picture (0/1) | Binary |
| nums/length username | Ratio of numbers in username | Float (0-1) |
| fullname words | Number of words in full name | Integer |
| nums/length fullname | Ratio of numbers in full name | Float (0-1) |
| name==username | Name matches username (0/1) | Binary |
| description length | Bio character count | Integer |
| external URL | Has external URL (0/1) | Binary |
| private | Account is private (0/1) | Binary |
| #posts | Number of posts | Integer |
| #followers | Number of followers | Integer |
| #follows | Number of following | Integer |

## Model Performance
| Model | Train Acc | Test Acc | CV Score | ROC AUC |
|-------|-----------|----------|----------|---------|
| Random Forest | 100.00% | 91.67% | 93.22% | 91.67% |
| SVM | 90.62% | 87.50% | 90.09% | 87.50% |
| Neural Network | 93.75% | 87.50% | — | 87.50% |

### Neural Network Architecture (from notebook)
```
Dense(50, relu) → Dropout(0.3) → Dense(150, relu) → Dropout(0.3) → Dense(25, relu) → Dropout(0.3) → Dense(2, softmax)
Optimizer: Adam | Loss: categorical_crossentropy | Epochs: 50
```

## Quick Start
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
# Register → Login → Dashboard → Detect
```

## Retrain Models
```bash
python train_models.py
```

## Project Structure
```
├── app.py                  # Flask application
├── train_models.py         # Model training script
├── requirements.txt        # Dependencies
├── data/
│   ├── train.csv           # Training dataset (576 rows)
│   └── test.csv            # Test dataset (120 rows)
├── models/
│   ├── random_forest.pkl   # Trained Random Forest
│   ├── svm.pkl             # Trained SVM
│   ├── neural_network.keras # Trained Neural Network
│   ├── scaler.pkl          # StandardScaler
│   └── results.json        # Accuracy comparison
└── templates/
    ├── base.html           # Base template (dark theme)
    ├── index.html          # Landing page
    ├── register.html       # User registration
    ├── login.html          # User login
    ├── dashboard.html      # Dashboard with model comparison
    ├── predict.html        # Prediction form (11 features)
    └── history.html        # Prediction history
```

## Sample Inputs for Testing

### Real Accounts (Expected: REAL)
| Profile Pic | Nums/Username | Name Words | Nums/Name | Name=User | Bio Len | URL | Private | Posts | Followers | Follows |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.27 | 0 | 0.0 | 0 | 53 | 0 | 0 | 32 | 1000 | 955 |
| 1 | 0.00 | 2 | 0.0 | 0 | 44 | 0 | 0 | 286 | 2740 | 533 |
| 1 | 0.00 | 2 | 0.0 | 0 | 82 | 0 | 1 | 319 | 328 | 668 |
| 1 | 0.00 | 1 | 0.0 | 0 | 143 | 0 | 1 | 273 | 14890 | 7369 |

### Fake Accounts (Expected: FAKE)
| Profile Pic | Nums/Username | Name Words | Nums/Name | Name=User | Bio Len | URL | Private | Posts | Followers | Follows |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0.62 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 4 | 72 |
| 0 | 0.50 | 1 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 |
| 0 | 0.40 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 18 | 44 |
| 1 | 0.00 | 1 | 0.0 | 1 | 0 | 0 | 0 | 0 | 2 | 360 |
