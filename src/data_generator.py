"""
Customer Churn Dataset Generator
Generates realistic Telecom / SaaS customer subscription and behavioral data
with realistic churn indicators for Decision Tree / Random Forest ML training.
"""

import os
import json
import random
import numpy as np
import pandas as pd

def generate_customer_churn_data(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset with non-linear relationships
    ideal for Decision Tree and Random Forest classifiers.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    customer_ids = [f"CUST-{1000 + i}" for i in range(num_samples)]
    genders = np.random.choice(["Male", "Female"], size=num_samples)
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=num_samples, p=[0.30, 0.70])
    
    # Tenure in months (1 to 72)
    tenure_months = np.random.randint(1, 73, size=num_samples)
    
    # Contract type
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=num_samples,
        p=[0.55, 0.24, 0.21]
    )
    
    # Internet service
    internet_services = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        size=num_samples,
        p=[0.44, 0.34, 0.22]
    )
    
    tech_supports = []
    online_securitiess = []
    online_backups = []
    device_protections = []
    streaming_tvs = []
    streaming_moviess = []
    monthly_chargess = []
    
    for i in range(num_samples):
        net = internet_services[i]
        if net == "No":
            tech_supports.append("No internet service")
            online_securitiess.append("No internet service")
            online_backups.append("No internet service")
            device_protections.append("No internet service")
            streaming_tvs.append("No internet service")
            streaming_moviess.append("No internet service")
            base_charge = np.random.uniform(18.5, 25.0)
        else:
            tech = np.random.choice(["Yes", "No"], p=[0.35, 0.65])
            sec = np.random.choice(["Yes", "No"], p=[0.30, 0.70])
            bak = np.random.choice(["Yes", "No"], p=[0.45, 0.55])
            prot = np.random.choice(["Yes", "No"], p=[0.40, 0.60])
            tv = np.random.choice(["Yes", "No"], p=[0.40, 0.60])
            mov = np.random.choice(["Yes", "No"], p=[0.40, 0.60])
            
            tech_supports.append(tech)
            online_securitiess.append(sec)
            online_backups.append(bak)
            device_protections.append(prot)
            streaming_tvs.append(tv)
            streaming_moviess.append(mov)
            
            base_charge = 50.0 if net == "DSL" else 75.0
            if tech == "Yes": base_charge += 8.0
            if sec == "Yes": base_charge += 7.0
            if bak == "Yes": base_charge += 6.0
            if tv == "Yes": base_charge += 10.0
            if mov == "Yes": base_charge += 10.0
            base_charge += np.random.uniform(-4.0, 6.0)
            
        monthly_chargess.append(round(base_charge, 2))
        
    paperless_billings = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=num_samples,
        p=[0.34, 0.22, 0.22, 0.22]
    )
    
    total_chargess = [
        round(m * t + np.random.uniform(-10.0, 10.0), 2)
        for m, t in zip(monthly_chargess, tenure_months)
    ]
    
    # Calculate churn probability based on non-linear rules (decision tree style ground truth)
    churn_labels = []
    for i in range(num_samples):
        score = 0.0
        # Rule 1: Contract type is major factor
        if contract_types[i] == "Month-to-month":
            score += 0.35
        elif contract_types[i] == "One year":
            score += 0.10
        else:
            score -= 0.15
            
        # Rule 2: Tenure duration
        if tenure_months[i] < 12:
            score += 0.30
        elif tenure_months[i] < 24:
            score += 0.10
        elif tenure_months[i] > 48:
            score -= 0.25
            
        # Rule 3: Fiber optic + High monthly charges + No tech support
        if internet_services[i] == "Fiber optic":
            score += 0.15
            if tech_supports[i] == "No":
                score += 0.20
            if online_securitiess[i] == "No":
                score += 0.15
                
        # Rule 4: Payment method
        if payment_methods[i] == "Electronic check":
            score += 0.12
            
        # Rule 5: Senior citizen without partner
        if senior_citizens[i] == 1 and partners[i] == "No":
            score += 0.10

        # Sigmoid activation with noise
        prob = 1.0 / (1.0 + np.exp(-3.5 * (score - 0.40)))
        # Convert to binary label
        churn_flag = 1 if np.random.rand() < prob else 0
        churn_labels.append(churn_flag)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": genders,
        "senior_citizen": senior_citizens,
        "partner": partners,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "contract_type": contract_types,
        "internet_service": internet_services,
        "tech_support": tech_supports,
        "online_security": online_securitiess,
        "online_backup": online_backups,
        "device_protection": device_protections,
        "streaming_tv": streaming_tvs,
        "streaming_movies": streaming_moviess,
        "paperless_billing": paperless_billings,
        "payment_method": payment_methods,
        "monthly_charges": monthly_chargess,
        "total_charges": total_chargess,
        "churn": churn_labels
    })

    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_customer_churn_data(1500)
    out_file = os.path.join(out_dir, "customer_churn.csv")
    df.to_csv(out_file, index=False)
    print(f"Generated synthetic customer churn dataset: {len(df)} records saved to {out_file}")
    print(f"Churn Rate: {df['churn'].mean() * 100:.2f}%")
