"""
Train & Evaluate Random Tree / Random Forest Model for Customer Churn Prediction
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC,
and saves model metadata & prediction results for the Web Dashboard.
Supports both Scikit-Learn and Pure-Python ML classifier backends.
"""

import os
import json
import random
import numpy as np
import pandas as pd

from src.random_tree_model import DecisionTreeClassifierCustom, RandomTreeForestClassifier, export_tree_to_json

# Try importing scikit-learn; fallback to pure python implementation if not yet installed
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def preprocess_churn_data(df: pd.DataFrame):
    """
    Encode categorical features, prepare feature matrix X and target y.
    Returns X_train, X_test, y_train, y_test, feature_names, encoders.
    """
    data = df.copy()
    
    categorical_cols = [
        "gender", "partner", "dependents", "contract_type",
        "internet_service", "tech_support", "online_security",
        "online_backup", "device_protection", "streaming_tv",
        "streaming_movies", "paperless_billing", "payment_method"
    ]
    
    numerical_cols = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]
    
    encoded_df = pd.DataFrame()
    encoders = {}
    
    for col in categorical_cols:
        unique_vals = sorted(data[col].unique().tolist())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_df[col] = data[col].map(mapping)
        encoders[col] = mapping
        
    for col in numerical_cols:
        encoded_df[col] = data[col].values
        
    X = encoded_df.values
    y = data["churn"].values
    feature_names = list(encoded_df.columns)
    
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # Pure Python Train/Test Split (80/20)
        n_samples = len(y)
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * 0.8)
        train_idxs, test_idxs = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idxs], X[test_idxs]
        y_train, y_test = y[train_idxs], y[test_idxs]
    
    return X_train, X_test, y_train, y_test, feature_names, encoders, encoded_df

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate basic classification performance metrics."""
    acc = np.mean(y_true == y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # Approximate ROC-AUC via rank sum / probability ordering
    sorted_pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        rank_sum = sum(i + 1 for i, (p, t) in enumerate(sorted_pairs) if t == 1)
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        
    cm = [[int(tn), int(fp)], [int(fn), int(tp)]]
    
    return float(acc), float(prec), float(rec), float(f1), float(auc), cm

def run_model_training_pipeline(csv_path: str, output_json_path: str):
    """Execute complete ML pipeline from dataset to evaluation metrics & model export."""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} records.")
    
    X_train, X_test, y_train, y_test, feature_names, encoders, encoded_df = preprocess_churn_data(df)
    
    if SKLEARN_AVAILABLE:
        print(" Using Scikit-Learn backend for Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_probas = rf_model.predict_proba(X_test)[:, 1]
        
        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_preds = dt_model.predict(X_test)
        dt_probas = dt_model.predict_proba(X_test)[:, 1]
        importances = rf_model.feature_importances_
    else:
        print(" Using Pure-Python Random Tree & Decision Forest Classifier engine...")
        rf_custom = RandomTreeForestClassifier(n_trees=15, max_depth=6)
        rf_custom.fit(X_train, y_train)
        rf_probas = rf_custom.predict_proba(X_test)
        rf_preds = (rf_probas >= 0.5).astype(int)
        
        dt_custom = DecisionTreeClassifierCustom(max_depth=5)
        dt_custom.fit(X_train, y_train)
        dt_probas = dt_custom.predict_proba(X_test)
        dt_preds = (dt_probas >= 0.5).astype(int)
        
        # Calculate feature importances from custom tree
        importances = np.zeros(len(feature_names))
        for feat_idx, val in dt_custom.feature_importances_.items():
            if feat_idx < len(feature_names):
                importances[feat_idx] = val
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

    # Train Custom Tree for export visual structure
    custom_dt = DecisionTreeClassifierCustom(max_depth=4)
    custom_dt.fit(X_train, y_train)
    tree_structure = export_tree_to_json(custom_dt.root, feature_names)

    # Compute Evaluation Metrics
    rf_acc, rf_prec, rf_rec, rf_f1, rf_auc, rf_cm = calculate_metrics(y_test, rf_preds, rf_probas)
    dt_acc, dt_prec, dt_rec, dt_f1, dt_auc, dt_cm = calculate_metrics(y_test, dt_preds, dt_probas)

    # Feature Importance Ranking
    feat_imp_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    
    # Sample Test Customer Predictions
    sample_customers = []
    test_indices = df.index[len(df) - len(y_test):].tolist()
    for idx, (real_idx, prob, pred, actual) in enumerate(zip(test_indices[:20], rf_probas[:20], rf_preds[:20], y_test[:20])):
        row = df.iloc[real_idx]
        sample_customers.append({
            "customer_id": row["customer_id"],
            "contract_type": row["contract_type"],
            "tenure_months": int(row["tenure_months"]),
            "monthly_charges": float(row["monthly_charges"]),
            "internet_service": row["internet_service"],
            "tech_support": row["tech_support"],
            "churn_prob": round(float(prob), 4),
            "predicted_churn": int(pred),
            "actual_churn": int(actual),
            "risk_level": "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")
        })

    output_data = {
        "metrics": {
            "random_forest": {
                "name": "Random Forest Ensemble",
                "accuracy": round(float(rf_acc), 4),
                "precision": round(float(rf_prec), 4),
                "recall": round(float(rf_rec), 4),
                "f1_score": round(float(rf_f1), 4),
                "roc_auc": round(float(rf_auc), 4),
                "confusion_matrix": rf_cm
            },
            "decision_tree": {
                "name": "Single Decision Tree",
                "accuracy": round(float(dt_acc), 4),
                "precision": round(float(dt_prec), 4),
                "recall": round(float(dt_rec), 4),
                "f1_score": round(float(dt_f1), 4),
                "roc_auc": round(float(dt_auc), 4),
                "confusion_matrix": dt_cm
            }
        },
        "feature_importance": feat_imp_list,
        "tree_structure": tree_structure,
        "sample_customers": sample_customers,
        "dataset_summary": {
            "total_customers": len(df),
            "churn_rate": round(float(df["churn"].mean() * 100), 2),
            "avg_tenure": round(float(df["tenure_months"].mean()), 1),
            "avg_monthly_charges": round(float(df["monthly_charges"].mean()), 2)
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n--- MODEL TRAINING & EVALUATION COMPLETE ---")
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}% | F1-Score: {rf_f1:.4f} | ROC-AUC: {rf_auc:.4f}")
    print(f"Decision Tree Accuracy: {dt_acc*100:.2f}% | F1-Score: {dt_f1:.4f} | ROC-AUC: {dt_auc:.4f}")
    print(f"Exported model metadata & metrics to: {output_json_path}\n")

    return output_data

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    csv_file = os.path.join(base_dir, "data", "customer_churn.csv")
    json_out = os.path.join(base_dir, "data", "model_outputs.json")
    run_model_training_pipeline(csv_file, json_out)
