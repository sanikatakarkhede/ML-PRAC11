from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# load files
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Spam Detection Model Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['message']
    
    transformed = vectorizer.transform([data])
    prediction = model.predict(transformed)[0]
    
    result = "Spam" if prediction == 1 else "Ham"
    
    return jsonify({'prediction': result})

# 🔥 IMPORTANT: deployment ke liye
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)