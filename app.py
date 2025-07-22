import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/reviews_dataset_5000.csv")
df = df.dropna(subset=["review", "label"])

# Feature engineering
df["review_length"] = df["review"].apply(len)
df["exclamation_count"] = df["review"].apply(lambda x: x.count("!"))

# Label encoding
df["label"] = df["label"].map({"genuine": 1, "fake": 0})

# Split
X = df[["review", "review_length", "exclamation_count"]]
y = df["label"]

# Vectorize text only
vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X["review"])

# Combine with numeric features
import scipy
X_full = scipy.sparse.hstack((X_vec, X[["review_length", "exclamation_count"]].values))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
