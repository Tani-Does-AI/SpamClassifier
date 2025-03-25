import os
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/tmp/nltk_data')
    nltk.data.path.append('/tmp/nltk_data')

# Rest of your imports
from flask import Flask, request, jsonify, render_template
import joblib
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data['email']
    cleaned_text = clean_text(email_text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)
    result = "spam" if prediction[0] == 1 else "ham"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)