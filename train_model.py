import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (download from Kaggle and place CSV here)
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")


df_fake["label"] = 0   # Fake
df_real["label"] = 1   # Real


df = pd.concat([df_fake, df_real])


df = df.sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df["text"] = df["text"].apply(clean_text)


X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved!")