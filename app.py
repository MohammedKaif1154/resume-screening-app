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

def calculate_ats_score(text):
    text = text.lower()

    skills = [
        "python", "sql", "excel", "machine learning", "data analysis",
        "tensorflow", "pandas", "numpy", "power bi", "tableau"
    ]

    sections = ["education", "experience", "projects", "skills"]

    # Skill Score (50)
    skill_matches = sum(1 for skill in skills if skill in text)
    skill_score = (skill_matches / len(skills)) * 50

    # Section Score (20)
    section_matches = sum(1 for sec in sections if sec in text)
    section_score = (section_matches / len(sections)) * 20

    # Keyword Density (15)
    words = text.split()
    unique_words = set(words)
    density = len(unique_words) / len(words) if len(words) > 0 else 0
    density_score = density * 15

    # Length Score (15)
    length_score = 15 if 300 <= len(words) <= 1200 else 5

    total_score = skill_score + section_score + density_score + length_score
    return round(total_score, 2)

def missing_skills(text):
    text = text.lower()

    skills = [
        "python", "sql", "excel", "machine learning",
        "pandas", "numpy", "power bi", "tableau"
    ]

    return [skill for skill in skills if skill not in text]


st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("📄 Resume Screening System")
st.write("Upload your resume and analyze it step-by-step.")


if "step" not in st.session_state:
    st.session_state.step = 1

if "text" not in st.session_state:
    st.session_state.text = ""


uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None and st.session_state.step == 1:

    st.success("Resume uploaded successfully ✅")

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

    # ML Prediction
    cleaned = clean_resume(text)
    vectorized = tfidf.transform([cleaned])

    prediction = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)[0]

    category = encoder.inverse_transform(prediction)

    st.success(f"🎯 ML Predicted Category: {category[0]}")

    # Domain Detection
    domain = detect_domain(text)
    st.info(f"🧠 Detected Domain: {domain}")

    # ATS Score
    ats_score = calculate_ats_score(text)

    st.subheader("📊 ATS Resume Score")
    st.metric("ATS Score", f"{ats_score}/100")

    if ats_score >= 80:
        st.success("Excellent resume! ✅")
    elif ats_score >= 60:
        st.warning("Good resume, but can be improved ⚠️")
    else:
        st.error("Resume needs improvement ❌")

    # Missing Skills
    missing = missing_skills(text)

    st.subheader("❗ Missing Skills Suggestions")
    st.write(missing[:5])

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

