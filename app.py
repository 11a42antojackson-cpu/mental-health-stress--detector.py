import streamlit as st
import joblib
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load saved model and vectorizer
model = joblib.load("model_rf.pkl") if False else joblib.load("best_model.pkl")  # use your logistic model if saved as best_model.pkl
tfidf = joblib.load("tfidf.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# Streamlit App UI
st.set_page_config(page_title="🧠 Stress Detection App", layout="wide")
st.title("🧠 Mental Health Stress Detection using NLP & ML")
st.write("Enter a text message or sentence below to check the stress level:")

user_input = st.text_area("📝 Enter your text here:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        clean_input = clean_text(user_input)
        vectorized_input = tfidf.transform([clean_input])
        prob = model.predict_proba(vectorized_input)[0][1]
        pred = 1 if prob > 0.7 else 0

        if pred == 1:
            st.error(f"😞 **Stressed** (Confidence: {prob*100:.2f}%)")
        else:
            st.success(f"😊 **Not Stressed** (Confidence: {(100 - prob*100):.2f}%)")

st.caption("⚙️ Model: Logistic Regression (Balanced) | Features: TF-IDF (Bigrams)")
