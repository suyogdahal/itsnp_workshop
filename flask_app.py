from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


# Load the saved CountVectorizer and MultinomialNB objects
with open("data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/classifier.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    __import__("pdb").set_trace()
    text = request.form["text"]

    # Vectorize the input text
    text_vec = vectorizer.transform([text]).toarray()

    # Use the trained model to make a prediction
    prediction = model.predict(text_vec)

    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
