import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Labeled Training Dataset (Balanced across Simple Account Queries vs Complex Compliance Queries)
queries = [
    # Simple Intent (Standard SQL / Governed KPI Lookup)
    "What is my current account balance?",
    "Show me my next mortgage payment date",
    "Update my billing address",
    "How many total loans are in UK_Retail?",
    "What is the total exposure in GBP for UK_Retail?",
    "Download my monthly mortgage statement",
    "What is my remaining loan principal?",
    "Check status of my direct debit",
    "Show total breaches count from the dashboard",
    "Reset my online banking password",
    "What is the average margin increase percentage?",
    "Give me my payment history for this year",
    
    # Complex Intent (Requires RAG + Contract Clause Reasoning)
    "Did the bank breach Clause 4.2 of loan LN-40291 by raising the margin by 0.75%?",
    "Analyze whether the early repayment charge on LN-88312 violates the agreement",
    "Does the commercial covenant in LN-99104 allow a 1.2% interest hike?",
    "Evaluate if the lender complied with notice period requirements in Clause 6",
    "Is a 0.75% margin increase within 12 months a contractual breach?",
    "Review the mortgage deed for any penalties on overpayment above 10%",
    "Did the bank violate variable rate cap rules under the loan contract?",
    "Check if the default interest rate charged exceeds the contractual ceiling",
    "Assess legal compliance of the margin adjustment on agreement LN-40291",
    "Can the lender demand immediate repayment under Clause 8 for late insurance?",
    "Compare the bank's 0.80% hike against the permitted commercial cap",
    "Determine if the borrower is entitled to a refund for an unlawful rate increase"
]

labels = ["simple"] * 12 + ["complex"] * 12

def train_and_evaluate_ml_router():
    os.makedirs("models", exist_ok=True)
    
    # 2. Feature Engineering: TF-IDF with Unigrams + Bigrams (n-gram range 1,2) and Sublinear TF scaling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )
    X = vectorizer.fit_transform(queries)
    y = np.array(labels)

    # 3. Model Benchmarking with 5-Fold Stratified Cross-Validation
    candidate_models = {
        "Logistic Regression (L2, Balanced)": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("=== 1. ML Model Benchmarking (5-Fold Stratified CV F1-Macro) ===")
    for name, model in candidate_models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
        print(f"• {name:35s} | Mean CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # 4. Hyperparameter Tuning on Logistic Regression via GridSearchCV
    param_grid = {
        "C": [0.1, 1.0, 5.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    }
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=42),
        param_grid,
        cv=skf,
        scoring="f1_macro"
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_

    print(f"\n=== 2. Best Hyperparameters Selected: {grid.best_params_} ===")
    y_pred = best_model.predict(X)
    # Probability of "complex" class for ROC-AUC
    complex_idx = list(best_model.classes_).index("complex")
    y_prob = best_model.predict_proba(X)[:, complex_idx]
    y_binary = (y == "complex").astype(int)

    print("\n=== 3. Classification Report ===")
    print(classification_report(y, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_binary, y_prob):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    # 5. Persist Artifacts for FastAPI
    joblib.dump(best_model, "models/router_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    print("\nSaved tuned ML router and TF-IDF vectorizer to 'models/'.")

if __name__ == "__main__":
    train_and_evaluate_ml_router()