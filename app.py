
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# --- Configuration ---
st.set_page_config(
    page_title="Comment Toxicity Detection",
    page_icon="🚫",
    layout="wide"
)

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_app_artifacts():
    """Loads the trained model and tokenizer."""
    try:
        model = load_model('best_toxicity_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

model, tokenizer = load_app_artifacts()

# --- Text Preprocessing ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans input text for prediction."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- Prediction Function ---
def predict_toxicity(comment):
    """Predicts the toxicity of a single comment."""
    if not model or not tokenizer:
        return None, None

    # Preprocess the comment
    cleaned_comment = clean_text(comment)
    
    # Vectorize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_comment])
    padded_sequence = pad_sequences(sequence, maxlen=150, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded_sequence, verbose=0)
    probability = prediction[0][0]
    
    return probability, probability > 0.5


# --- UI Layout ---
st.title("🚫 Deep Learning for Comment Toxicity Detection")
st.markdown("""
This application uses a Bidirectional LSTM model to predict whether a comment is toxic. 
Enter a comment in the text box below or upload a CSV file for bulk prediction.
""")

# --- Single Comment Prediction ---
st.header("Real-Time Toxicity Analysis")
user_input = st.text_area("Enter a comment to analyze:", height=150, placeholder="Type your comment here...")

if st.button("Analyze Comment"):
    if user_input:
        with st.spinner("Analyzing..."):
            probability, is_toxic = predict_toxicity(user_input)
            if probability is not None:
                st.subheader("Analysis Result")
                if is_toxic:
                    st.error(f"**Toxic Comment Detected** (Probability: {probability:.2%})")
                else:
                    st.success(f"**Non-Toxic Comment** (Probability of being toxic: {probability:.2%})")
    else:
        st.warning("Please enter a comment to analyze.")

# --- Bulk Prediction ---
st.header("Bulk Prediction with CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Let user select the column with comments
        comment_column = st.selectbox("Select the column containing comments:", df.columns)

        if st.button("Predict on CSV"):
            with st.spinner("Processing file... This may take a while for large files."):
                predictions = df[comment_column].apply(lambda x: predict_toxicity(str(x)))
                df['toxicity_probability'] = [p[0] for p in predictions]
                df['is_toxic'] = [p[1] for p in predictions]
                
                st.success("Prediction complete!")
                st.dataframe(df)

                # Provide download link
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='toxicity_predictions.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
