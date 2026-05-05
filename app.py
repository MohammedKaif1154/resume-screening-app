import streamlit as st
import pickle
import re
import pandas as pd
import pdfplumber

# Load saved model files
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("clf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
# Clean function
def clean_resume(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text) 
    text = re.sub(r'\s+', ' ', text) 
    return text.lower()

def detect_domain(text):
    text = text.lower()

    domains = {
        "Data Science": ["python", "machine learning", "pandas", "numpy", "data analysis", "model"],
        "Web Development": ["html", "css", "javascript", "react", "node", "frontend", "backend"],
        "Android Development": ["android", "kotlin", "java", "xml"],
        "HR": ["recruitment", "hiring", "onboarding", "talent"],
        "Finance": ["accounting", "finance", "tax", "audit"],
        "DevOps": ["docker", "kubernetes", "aws", "ci cd", "pipeline"],
        "Testing": ["testing", "selenium", "automation", "qa"],
    }

    scores = {}

    for domain, keywords in domains.items():
        score = 0
        for word in keywords:
            if word in text:
                score += 1
        scores[domain] = score

    return max(scores, key=scores.get)

st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("📄 Resume Screening System")
st.write("Upload your resume (PDF) and get instant job category prediction.")

# 🔹 Drag & Drop PDF
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                
    st.subheader("Extracted Resume Text (Preview)")
    st.write(text[:1000])
    
   if uploaded_file is not None:

    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    st.subheader("Extracted Resume Text (Preview)")
    st.write(text[:1000])

    # 👇 EVERYTHING BELOW MUST BE INSIDE BUTTON
    if st.button("Analyze Resume"):

        cleaned = clean_resume(text)
        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)
        probabilities = model.predict_proba(vectorized)[0]

        category = encoder.inverse_transform(prediction)

        st.success(f"ML Predicted Category: {category[0]}")

        # ✅ Keyword detection
        domain = detect_domain(text)
        st.info(f"Detected Domain: {domain}")

        # ✅ Confidence chart (NOW SAFE)
        proba_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Confidence": probabilities
        }).sort_values(by="Confidence", ascending=False)

        st.subheader("Prediction Confidence")
        st.bar_chart(proba_df.set_index("Category").head(5))
