import pathlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "resume_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
CONFUSION_MATRIX_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"
CATEGORY_DIST_PATH = BASE_DIR / "outputs" / "category_distribution.png"


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} resumes across {df['category'].nunique()} categories")

    # --- Data insight #1: category distribution (save as a chart your app can show) ---
    plt.figure(figsize=(8, 5))
    df["category"].value_counts().plot(kind="barh", color="#4C72B0")
    plt.title("Training Data: Resumes per Category")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(CATEGORY_DIST_PATH)
    plt.close()
    print(f"Saved category distribution chart -> {CATEGORY_DIST_PATH}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["resume_text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    # --- Vectorize text ---
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Train model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    print("\n=== Classification Report ===")
    print(report)

    # --- Data insight #2: confusion matrix ---
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Resume Category Classifier")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Saved confusion matrix -> {CONFUSION_MATRIX_PATH}")

    # --- Save model + vectorizer so the app can load them without retraining ---
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
