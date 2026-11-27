import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "symptoms.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "health_model.pkl")


def train_model():

    print("Loading dataset...")

    data = pd.read_csv(DATASET_PATH)

    data.dropna(subset=["symptoms", "condition"], inplace=True)

    X = data["symptoms"].astype(str)
    y = data["condition"].astype(str)

    print("Dataset loaded successfully!")
    print(f"Total records: {len(data)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTraining machine learning model...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print("-------------------------")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel successfully trained!")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()