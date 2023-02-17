import streamlit as st
import pickle

# Load the vectorizer and classifier from pickle files
with open("data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Define a function to make predictions
def predict(text: str) -> str:
    # Vectorize the input text and predict
    vectorized_text = vectorizer.transform([text]).toarray()
    return classifier.predict(vectorized_text)


# Define the Streamlit app
def app():
    st.title("Simple text classification app")

    # Add a text input field
    text = st.text_input("Enter some text")

    # Add a button to make predictions
    if st.button("Classify"):
        prediction = predict(text)
        st.write("The sentence's sentiment is:", prediction)


# Run the app
if __name__ == "__main__":
    app()
