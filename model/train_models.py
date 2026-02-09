"""
Wine Quality Classification - Model Training
Author: Prithvi
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


def load_wine_data():
    """Load and prepare the wine quality dataset"""
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    try:
        red = pd.read_csv(red_url, sep=';')
        white = pd.read_csv(white_url, sep=';')
        red['wine_type'] = 0
        white['wine_type'] = 1
        data = pd.concat([red, white], ignore_index=True)
    except:
        print("Could not fetch from UCI, trying local file...")
        data = pd.read_csv('winequality.csv')
    
    # Binary classification: good wine (quality >= 6) vs bad wine
    data['label'] = (data['quality'] >= 6).astype(int)
    
    X = data.drop(['quality', 'label'], axis=1)
    y = data['label']
    
    return X, y


def calc_metrics(y_true, y_pred, y_prob):
    """Calculate all required evaluation metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }


def train_models(X_train, X_test, y_train, y_test, scaler):
    """Train all 6 classification models"""
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
    }
    
    # Models that need scaled data
    needs_scaling = ['Logistic Regression', 'KNN']
    
    results = {}
    trained = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name in needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = calc_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        trained[name] = model
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}, F1: {metrics['F1']:.4f}")
    
    return trained, results


def save_artifacts(models, scaler, results, features):
    """Save models and other artifacts"""
    os.makedirs('model', exist_ok=True)
    
    for name, model in models.items():
        fname = name.lower().replace(' ', '_') + '.pkl'
        with open(f'model/{fname}', 'wb') as f:
            pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('model/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    print("Wine Quality Classification - Training Script")
    print("-" * 50)
    
    # Load data
    print("\nLoading dataset...")
    X, y = load_wine_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples\n")
    scaler = StandardScaler()
    models, results = train_models(X_train, X_test, y_train, y_test, scaler)
    
    # Save
    save_artifacts(models, scaler, results, list(X.columns))
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    df = pd.DataFrame(results).T.round(4)
    print(df.to_string())
    
    # Save for README
    df.to_csv('model/comparison_results.csv')
    
    # Save test data for app
    test_df = X_test.copy()
    test_df['actual_label'] = y_test.values
    test_df.to_csv('model/test_data.csv', index=False)
    
    print("\nDone! Models saved to 'model/' folder.")
