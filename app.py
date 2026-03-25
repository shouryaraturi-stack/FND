from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("CLASSES:", model.classes_)  # Debug

# Load stopwords
stop_words = set(stopwords.words('english'))

# Clean text (same as training)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(text):
    # Input validation
    if len(text.split()) < 5:
        return "⚠️ Please enter a proper news sentence (at least 5 words)"

    # Clean text
    text = clean_text(text)

    # Transform
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]

    print("PROBABILITIES:", prob)  # Debug

    confidence = max(prob)

    # Prediction logic
    if prob[1] > prob[0]:   # 1 = REAL
        result = "🟢 Real News"
    else:
        result = "🔴 Fake News"

    result += f" ({confidence*100:.2f}% sure)"

    return result

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    print("USER INPUT:", news)

    result = predict_news(news)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)