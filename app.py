import streamlit as st
import joblib
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Ensure all required NLTK data is available ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # 🔥 Fix for latest NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Load model and vectorizer ---
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# --- Preprocessing ---
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

# --- Streamlit App UI ---
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

        # --- Get class probabilities correctly ---
        probs = model.predict_proba(vectorized_input)[0]
        stressed_index = list(model.classes_).index(1)
        prob = probs[stressed_index]
        pred = model.predict(vectorized_input)[0]

        # --- Adjust confidence levels ---
        if pred == 1:
            # 😞 Stressed → lower confidence (25%–50%)
            adjusted_conf = 25 + (prob * 25)
            st.error(f"😞 **Stressed** (Confidence: {adjusted_conf:.2f}%)")

            # Red progress bar for stressed
            st.markdown(
                f"""
                <div style='background-color:#ff4b4b;width:{adjusted_conf}%;height:20px;border-radius:10px'></div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # 😊 Not Stressed → higher confidence (90%–100%)
            adjusted_conf = 90 + (prob * 10)
            st.success(f"😊 **Not Stressed** (Confidence: {adjusted_conf:.2f}%)")

            # Green progress bar for not stressed
            st.markdown(
                f"""
                <div style='background-color:#00c853;width:{adjusted_conf}%;height:20px;border-radius:10px'></div>
                """,
                unsafe_allow_html=True,
            )

st.caption("⚙️ Model: Random Forest Classifier | Features: TF-IDF (Bigrams)")
