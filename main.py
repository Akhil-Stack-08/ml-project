"""
Customer Churn Prediction using Random Tree / Random Forest
Main entry point for pipeline execution:
1. Dataset Generation
2. Machine Learning Training & Hyperparameter Evaluation
3. Tree Logic & Rule Extraction
4. JSON Export for Interactive Web Dashboard
"""

import os
import sys
from src.data_generator import generate_customer_churn_data
from src.train_eval import run_model_training_pipeline

def main():
    print("=" * 65)
    print(" CUSTOMER CHURN PREDICTION USING RANDOM TREE / RANDOM FOREST ")
    print("=" * 65)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_file = os.path.join(data_dir, "customer_churn.csv")
    json_out = os.path.join(data_dir, "model_outputs.json")
    
    print("\n[Step 1] Generating synthetic customer churn dataset...")
    df = generate_customer_churn_data(num_samples=1500, random_seed=42)
    df.to_csv(csv_file, index=False)
    print(f" Saved dataset to: {csv_file}")
    print(f" Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f" Base Customer Churn Rate: {df['churn'].mean() * 100:.2f}%\n")
    
    print("[Step 2] Training Decision Tree & Random Forest ML Classifiers...")
    results = run_model_training_pipeline(csv_file, json_out)
    
    print("[Step 3] Summary Metrics:")
    rf_m = results["metrics"]["random_forest"]
    dt_m = results["metrics"]["decision_tree"]
    
    print(f" -> Random Forest Classifier  | Accuracy: {rf_m['accuracy']*100:.2f}% | Precision: {rf_m['precision']:.4f} | Recall: {rf_m['recall']:.4f} | ROC-AUC: {rf_m['roc_auc']:.4f}")
    print(f" -> Single Decision Tree     | Accuracy: {dt_m['accuracy']*100:.2f}% | Precision: {dt_m['precision']:.4f} | Recall: {dt_m['recall']:.4f} | ROC-AUC: {dt_m['roc_auc']:.4f}")
    
    print("\n[Step 4] Top 5 Most Important Features for Churn:")
    for item in results["feature_importance"][:5]:
        print(f"   - {item['feature']:<20}: {item['importance']*100:.2f}% relative importance")
        
    print("\n" + "=" * 65)
    print(" Pipeline complete! You can launch the interactive web dashboard")
    print(" by opening 'index.html' in your browser.")
    print("=" * 65)

if __name__ == "__main__":
    main()
