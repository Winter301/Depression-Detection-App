import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required libraries are installed
try:
    import matplotlib
    import seaborn
    import sklearn
    import wordcloud
except ImportError:
    raise ImportError("Make sure to install required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, wordcloud")

# Streamlit app title
st.title("Depression Detection App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    try:
        df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        st.write("Dataset Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading the file: {e}")

    # Load the pre-trained model and vectorizer
    try:
        with open('best_svc_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError as e:
        st.error("Model files not found. Please ensure 'best_svc_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as the app.")
    except Exception as e:
        st.error(f"Error loading model files: {e}")

    # Instructions for the user
    st.write("This app uses a machine learning model to predict if a given text indicates depression. Please enter a sentence or text in the input box below.")

    # User input for text
    user_input = st.text_area("Enter your sentence or words:", height=150)

    # Prediction button
    if st.button("Predict"):
        if user_input.strip():
            try:
                # Transform the input using the loaded vectorizer
                input_vectorized = vectorizer.transform([user_input]).toarray()
                
                # Make prediction
                prediction = model.predict(input_vectorized)
                
                # Display the result
                if prediction[0] == 1:
                    st.error("The model predicts: You may be experiencing depression. Please consider reaching out to a mental health professional for support.")
                else:
                    st.success("The model predicts: You are not experiencing depression. Stay positive and take care of your mental health!")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

# Footer
st.write("\n---\n")
st.caption("Disclaimer: This app is for informational purposes only and is not a substitute for professional mental health advice. If you are feeling distressed, please seek help from a licensed professional.")
