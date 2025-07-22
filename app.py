import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("reviews_dataset_5000.csv")

# Drop rows with null review or label
df = df.dropna(subset=["review", "label"])

# Encode labels: genuine = 1, fake = 0
df["label"] = df["label"].map({"genuine": 1, "fake": 0})

# Split into features and target
X = df["review"]
y = df["label"]

# Define pipeline: CountVectorizer + Logistic Regression
model = Pipeline([
    ("vectorizer", CountVectorizer(max_features=3000)),
    ("classifier", LogisticRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model to disk
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
