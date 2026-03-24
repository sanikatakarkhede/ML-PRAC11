# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ✅ Load dataset (LOCAL FILE)
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print(data.head())

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
import pickle

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))