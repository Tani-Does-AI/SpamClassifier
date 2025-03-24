import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.corpus import stopwords
import string
import os
import nltk


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Download stopwords
nltk.download('stopwords')

# Load and clean data
def load_and_clean_data():
    df = pd.read_csv('email.csv', encoding='latin-1')
    df.columns = ['Category', 'Message']
    
    # Clean data
    df = df.dropna(subset=['Category', 'Message'])
    df['Category'] = df['Category'].str.strip().str.lower()
    df = df[df['Category'].isin(['ham', 'spam'])]
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    
    print(f"Final dataset size: {len(df)}")
    print("Label distribution:")
    print(df['Category'].value_counts())
    
    return df

df = load_and_clean_data()

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Message'] = df['Message'].apply(clean_text)

# Train-test split
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify no NaN in y_train
print("NaN in y_train:", y_train.isnull().sum())

# Vectorize and train
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save artifacts
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model saved successfully!")