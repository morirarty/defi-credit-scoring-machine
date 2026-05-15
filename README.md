# 🏦 DeFi Credit Scoring: AI-Powered Risk Classification

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Dune Analytics](https://img.shields.io/badge/Dune_API-000000?style=for-the-badge&logo=dune&logoColor=white)

## 📌 Executive Summary
In Traditional Finance (TradFi), credit scores dictate capital efficiency. In Web3, wallet anonymity forces lending protocols (like Aave) to rely on highly capital-inefficient overcollateralization. 

This project bridges that gap by building a **Machine Learning Classification Model** that analyzes a wallet's on-chain footprint to predict its likelihood of default (liquidation). By transitioning from strict collateral rules to behavioral credit scoring, Web3 protocols can safely move towards undercollateralized lending.

## 🏗️ Data Architecture & Extraction
Instead of relying on aggregated community tables which can be prone to schema changes, this pipeline extracts fundamental truth directly from **Raw EVM Event Logs**:
* Queried Aave V2 and V3 `LiquidationCall` smart contract events via **Dune Analytics**.
* Extracted the exact `user` parameter from the Solidity logs to identify defaulted wallets.
* Engineered on-chain behavioral features mapping to each borrower: `total_transactions` and `wallet_age_days`.

## 🧠 Machine Learning Pipeline
Built using Python and `scikit-learn`, addressing the specific challenges of Web3 data:
* **Algorithm:** Implemented a **Random Forest Classifier** to categorize wallets into "Safe" (0) or "High Risk" (1).
* **Imbalanced Data Handling:** Applied `class_weight='balanced'` to penalize the AI heavily for missing rare liquidation events, prioritizing protocol safety over raw accuracy.
* **Fair Validation:** Utilized `stratify=y` during the train/test split to ensure the validation dataset contained a realistic proportion of defaulted wallets.

## 📊 Key Business Insight: The "Sybil-Resistant" Feature
<img width="790" height="590" alt="image (9)" src="https://github.com/user-attachments/assets/4c9705b6-dd2f-45ef-af54-ea02d1a97357" />


Upon analyzing the model's **Feature Importance**, a critical Web3 behavioral pattern emerged:
* **Total Transactions:** 25.7% importance
* **Wallet Age (Days):** 74.3% importance

**The Alpha:** While malicious actors can easily write scripts to spam 10,000 transactions in a day to artificially inflate a wallet's "activity score," **time cannot be spoofed**. Older wallets demonstrate survival through multiple market cycles (bull and bear) and indicate experienced risk management (maintaining healthy Loan-to-Value ratios). The AI independently discovered this Sybil-resistant logic purely through mathematical variance.

---
