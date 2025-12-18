import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -------------------- NLTK SETUP (CLOUD SAFE) --------------------
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

setup_nltk()


# -------------------- LOAD MODEL & VECTORIZER --------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()


# -------------------- TEXT PREPROCESSING --------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    cleaned = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned.append(ps.stem(word))

    return " ".join(cleaned)


# -------------------- STREAMLIT UI --------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.write("Detect whether a message is **Spam** or **Not Spam** using ML.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)

        # TF-IDF gives sparse output (KEEP IT SPARSE)
        vector_input = vectorizer.transform([transformed_sms]).toarray()

        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")
