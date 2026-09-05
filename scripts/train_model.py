# scripts/train_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def train_and_evaluate():
    """
    Loads the pairwise edge dataset, trains a Random Forest classifier to 
    predict structural conflicts (label 1) vs safe combinations (label 0),
    and saves the trained model for Phase 3.
    """
    
    # ---------------------------------------------------------
    # 1. Path Definitions & Data Loading
    # ---------------------------------------------------------
    # Ensure relative paths work regardless of where the script is executed from
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'scripts', 'datasets', 'training_data.csv')
    model_dir = os.path.join(base_dir, 'src', 'pagerankfas', 'ml_enhanced')
    model_path = os.path.join(model_dir, 'rf_model.pkl')

    print(f"[*] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    
    print("\n[*] Dataset Label Distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    # ---------------------------------------------------------
    # 2. Data Splitting
    # ---------------------------------------------------------
    # Separate features (X) and target label (y)
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # We use stratify=y to ensure the 72% / 28% class distribution 
    # remains consistent in both training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[*] Training set size: {len(X_train)} pairs")
    print(f"[*] Testing set size: {len(X_test)} pairs")

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    print("\n[*] Training RandomForestClassifier with balanced class weights...")
    # n_jobs=-1 uses all available CPU cores for training
    # class_weight="balanced" forces the model to heavily penalize missing Class 1
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight="balanced" 
    )
    rf_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. Evaluation Metrics
    # ---------------------------------------------------------
    # Get raw conflict probabilities instead of hard predictions
    y_probs = rf_model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("        DECISION THRESHOLD ANALYSIS (COST-SENSITIVE)")
    print("="*60)

    thresholds = [0.50, 0.40, 0.35, 0.30]
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n--- Threshold: {thresh:.2f} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"False Negatives (Dangerous Misses) : {cm[1][0]}")
        print(f"True Positives  (Caught Conflicts) : {cm[1][1]}")

    print("\n" + "="*60)

    # ---------------------------------------------------------
    # 5. Feature Importances
    # ---------------------------------------------------------
    print("\n--- Feature Importances ---")
    importances = sorted(zip(X.columns, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
    
    for feature, imp in importances:
        print(f"{feature:<20}: {imp:.4f}")

    # ---------------------------------------------------------
    # 6. Save Model
    # ---------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, model_path)
    print(f"\n[*] Model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()