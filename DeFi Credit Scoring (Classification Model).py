# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.6",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "requests==2.33.1",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    import warnings
    warnings.filterwarnings('ignore')

    # ==========================================
    # 1. DUNE API DATA INGESTION
    # ==========================================
    QUERY_ID = '7500922' 
    API_KEY = 'iEuaSNBZeMLZcwoqQdAqWa1uVUgyQ8d3'

    url = f"https://api.dune.com/api/v1/query/{7500922}/results"
    headers = {"x-dune-api-key": "iEuaSNBZeMLZcwoqQdAqWa1uVUgyQ8d3"}

    print("Fetching wallet behavior data from Dune API...")
    response = requests.get(url, headers=headers)
    data = response.json()

    if 'result' in data and 'rows' in data['result']:
        df = pd.DataFrame(data['result']['rows'])
        print(f"Success! Loaded {len(df)} wallets for credit analysis.")
    else:
        raise ValueError(f"API Error. Check keys/ID. Log: {data}")

    # ==========================================
    # 2. DATA PREPARATION
    # ==========================================
    # Ensure data types are correct
    df['total_transactions'] = pd.to_numeric(df['total_transactions'])
    df['wallet_age_days'] = pd.to_numeric(df['wallet_age_days'])
    df['target_default'] = pd.to_numeric(df['target_default'])

    # Define Features (X) and Target Label (y)
    X = df[['total_transactions', 'wallet_age_days']]
    y = df['target_default']

    # ==========================================
    # 3. MACHINE LEARNING (Classification)
    # ==========================================
    print("Training the Credit Scoring AI...")

    # Split Data: 80% Training, 20% Testing
    # Note: We use stratify=y to ensure the 20% test data has a fair ratio of good vs bad borrowers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Random Forest Classifier
    # class_weight='balanced' is CRITICAL because liquidations (1) are rare compared to safe borrowers (0)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    predictions = model.predict(X_test)

    # ==========================================
    # 4. EVALUATION & VISUALIZATION (Confusion Matrix)
    # ==========================================
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, predictions, target_names=['Safe (0)', 'High Risk / Liquidated (1)']))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a beautiful heatmap for the Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Safe', 'Predicted Risk'], 
                yticklabels=['Actual Safe', 'Actual Risk'],
                annot_kws={"size": 16, "weight": "bold"})

    plt.title('DeFi Credit Scoring: AI Risk Detection', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Machine Prediction', fontsize=12, color='gray')
    plt.ylabel('Actual On-Chain Reality', fontsize=12, color='gray')

    fig.tight_layout()
    plt.show()

    # ==========================================
    # 5. BONUS: FEATURE IMPORTANCE
    # ==========================================
    # Let's see which feature the AI thinks is more important for credit scoring
    importances = model.feature_importances_
    print(f"\n--- Machine Decision Logic ---")
    print(f"Weight of 'Total Transactions': {importances[0]*100:.1f}%")
    print(f"Weight of 'Wallet Age (Days)': {importances[1]*100:.1f}%")
    return


if __name__ == "__main__":
    app.run()
