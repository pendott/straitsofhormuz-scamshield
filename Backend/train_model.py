import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

print("Original columns:", df.columns.tolist())

# Handle common Kaggle SMS spam CSV format
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

# Handle already-clean format
elif "label" in df.columns and "message" in df.columns:
    df = df[["label", "message"]].copy()

else:
    raise ValueError(
        "Could not find expected columns. Need either ['v1','v2'] or ['label','message']."
    )

print("Cleaned columns:", df.columns.tolist())
print(df.head())

# Convert labels
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Drop bad rows
df = df.dropna(subset=["message", "label_num"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)

# Build model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "scam_model.pkl")
print("Model saved as scam_model.pkl")