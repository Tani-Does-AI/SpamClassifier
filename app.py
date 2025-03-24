from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function (must match training preprocessing)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data['email']
    
    # Clean and vectorize the input
    cleaned_text = clean_text(email_text)
    text_vec = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vec)
    result = "spam" if prediction[0] == 1 else "ham"
    
    return jsonify({"result": result})

# Serve frontend
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)