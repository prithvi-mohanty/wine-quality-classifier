"""
Wine Quality Prediction App
Author: Prithvi
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# Model info
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

NEEDS_SCALING = ['Logistic Regression', 'KNN']


@st.cache_resource
def load_model(name):
    """Load a trained model"""
    path = f'model/{MODEL_FILES[name]}'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_scaler():
    """Load the scaler"""
    if os.path.exists('model/scaler.pkl'):
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_feature_names():
    """Load feature names"""
    if os.path.exists('model/feature_names.pkl'):
        with open('model/feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_results():
    """Load saved results"""
    if os.path.exists('model/results.pkl'):
        with open('model/results.pkl', 'rb') as f:
            return pickle.load(f)
    return None


def calc_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    return fig


def main():
    st.title("🍷 Wine Quality Prediction")
    st.write("ML Assignment 2 - Classification Models | Prithvi - BITS Pilani WILP")
    
    # Sidebar - Model selection
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Dataset Info**")
    st.sidebar.write("Wine Quality Dataset (UCI)")
    st.sidebar.write("- 12 Features")
    st.sidebar.write("- 6497 Samples")
    st.sidebar.write("- Binary Classification")
    
    # Load resources
    model = load_model(selected_model)
    scaler = load_scaler()
    feature_names = load_feature_names()
    saved_results = load_results()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Evaluate", "📊 Compare", "ℹ️ About"])
    
    with tab1:
        st.header("Upload & Evaluate")
        
        # File upload
        uploaded_file = st.file_uploader("Upload test CSV", type=['csv'])
        use_sample = st.checkbox("Use sample test data", value=True)
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
            use_sample = False
        elif use_sample and os.path.exists('model/test_data.csv'):
            data = pd.read_csv('model/test_data.csv')
            st.info("Using sample test data")
        else:
            data = None
        
        if data is not None:
            with st.expander("Preview Data"):
                st.dataframe(data.head(10))
                st.write(f"Shape: {data.shape}")
            
            if 'actual_label' not in data.columns:
                st.error("CSV must have 'actual_label' column")
                st.stop()
            
            y_true = data['actual_label'].values
            X = data.drop('actual_label', axis=1)
            
            if feature_names:
                X = X[feature_names]
            
            if model:
                st.subheader(f"Results: {selected_model}")
                
                # Scale if needed
                if selected_model in NEEDS_SCALING and scaler:
                    X_proc = scaler.transform(X)
                else:
                    X_proc = X.values
                
                y_pred = model.predict(X_proc)
                y_prob = model.predict_proba(X_proc)[:, 1]
                
                metrics = calc_metrics(y_true, y_pred, y_prob)
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                col2.metric("AUC", f"{metrics['AUC']:.4f}")
                col3.metric("Precision", f"{metrics['Precision']:.4f}")
                col4.metric("Recall", f"{metrics['Recall']:.4f}")
                col5.metric("F1", f"{metrics['F1']:.4f}")
                col6.metric("MCC", f"{metrics['MCC']:.4f}")
                
                st.markdown("---")
                
                # Confusion matrix and report
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("Confusion Matrix")
                    fig = plot_confusion_matrix(y_true, y_pred, selected_model)
                    st.pyplot(fig)
                
                with col_right:
                    st.subheader("Classification Report")
                    report = classification_report(y_true, y_pred, 
                                                   target_names=['Bad', 'Good'],
                                                   output_dict=True)
                    st.dataframe(pd.DataFrame(report).T.round(4))
            else:
                st.error(f"Model not found. Run train_models.py first.")
    
    with tab2:
        st.header("Model Comparison")
        
        if saved_results:
            df = pd.DataFrame(saved_results).T.round(4)
            
            st.subheader("Metrics Table")
            st.dataframe(df.style.highlight_max(axis=0, color='lightgreen')
                                 .highlight_min(axis=0, color='lightcoral'))
            
            st.markdown("---")
            
            # Bar charts
            st.subheader("Visual Comparison")
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            for idx, (ax, metric) in enumerate(zip(axes.flatten(), metrics_list)):
                values = df[metric].values
                models = df.index.tolist()
                ax.barh(models, values, color=plt.cm.Blues(0.5 + idx*0.08))
                ax.set_xlabel(metric)
                ax.set_xlim(0, 1.05)
                for i, v in enumerate(values):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best models
            st.markdown("---")
            st.subheader("Best Models")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**Best Accuracy:** {df['Accuracy'].idxmax()} ({df['Accuracy'].max():.4f})")
            c2.success(f"**Best AUC:** {df['AUC'].idxmax()} ({df['AUC'].max():.4f})")
            c3.success(f"**Best F1:** {df['F1'].idxmax()} ({df['F1'].max():.4f})")
        else:
            st.warning("No results found. Run train_models.py first.")
    
    with tab3:
        st.header("About")
        
        st.markdown("""
        ### Problem Statement
        Predict wine quality based on chemical properties. Binary classification where 
        wines with quality ≥ 6 are "Good" and < 6 are "Bad".
        
        ### Dataset
        Wine Quality dataset from UCI ML Repository containing red and white wine samples.
        
        **Features (12):**
        - fixed acidity, volatile acidity, citric acid
        - residual sugar, chlorides
        - free sulfur dioxide, total sulfur dioxide
        - density, pH, sulphates, alcohol
        - wine_type (0=red, 1=white)
        
        ### Models
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors (k=5)
        4. Gaussian Naive Bayes
        5. Random Forest (100 trees)
        6. XGBoost
        
        ### Metrics
        Accuracy, AUC, Precision, Recall, F1 Score, MCC
        
        ---
        *M.Tech AIML - BITS Pilani WILP*
        """)


if __name__ == "__main__":
    main()
