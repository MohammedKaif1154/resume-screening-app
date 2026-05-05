import streamlit as st
import pickle
import re
import pandas as pd
import pdfplumber

tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("clf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

def clean_resume(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def detect_domain(text):
    text = text.lower()

    domains = {
        "Data Science": ["python", "machine learning", "pandas", "numpy", "data analysis"],
        "Web Development": ["html", "css", "javascript", "react", "node"],
        "Android Development": ["android", "kotlin", "java"],
        "HR": ["recruitment", "hiring", "onboarding"],
        "Finance": ["accounting", "finance", "tax", "audit"],
        "DevOps": ["docker", "kubernetes", "aws"],
        "Testing": ["testing", "selenium", "qa"],
    }

    scores = {}
    for domain, keywords in domains.items():
        scores[domain] = sum(1 for word in keywords if word in text)

    return max(scores, key=scores.get)

st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("📄 Resume Screening System")
st.write("Upload your resume (PDF) and analyze it step-by-step.")

if "step" not in st.session_state:
    st.session_state.step = 1

if "text" not in st.session_state:
    st.session_state.text = ""

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None and st.session_state.step == 1:
    st.success("Resume uploaded successfully ✅")

    # Extract text once and store
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    st.session_state.text = text

    if st.button("Next ➡️"):
        st.session_state.step = 2

if st.session_state.step >= 2 and st.session_state.text:

    st.subheader("📄 Resume Preview")
    st.write(st.session_state.text[:1000])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1

    with col2:
        if st.button("Analyze ➡️"):
            st.session_state.step = 3


if st.session_state.step == 3:

    text = st.session_state.text

    cleaned = clean_resume(text)
    vectorized = tfidf.transform([cleaned])

    prediction = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)[0]

    category = encoder.inverse_transform(prediction)

    st.success(f"🎯 ML Predicted Category: {category[0]}")

    # Keyword-based domain
    domain = detect_domain(text)
    st.info(f"🧠 Detected Domain (Keyword-Based): {domain}")

    # Confidence Chart
    proba_df = pd.DataFrame({
        "Category": encoder.classes_,
        "Confidence": probabilities
    }).sort_values(by="Confidence", ascending=False)

    st.subheader("📊 Prediction Confidence")
    st.bar_chart(proba_df.set_index("Category").head(5))

    if st.button("✅ Done"):
        st.session_state.step = 4

if st.session_state.step == 4:
    st.success("Analysis Completed 🎉")

    if st.button("🔄 Analyze Another Resume"):
        st.session_state.step = 1
        st.session_state.text = ""

st.progress(st.session_state.step / 4)
