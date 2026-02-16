📄  Resume Screening System
🚀 Live Demo

🔗 Deployed App:(https://kaif-resume-screening.streamlit.app)
🔗 GitHub Repository:(https://github.com/MohammedKaif1154/resume-screening-app)

📌 Project Overview

The AI Resume Screening System is an end-to-end Machine Learning application that automatically classifies resumes into job categories using Natural Language Processing (NLP).

The system allows users to upload a resume (PDF format), extracts the text, processes it using TF-IDF vectorization, and predicts the most relevant job category using a trained Logistic Regression model.

This project simulates how modern HR systems and Applicant Tracking Systems (ATS) analyze resumes at scale.

🎯 Problem Statement

Recruiters receive hundreds of resumes for various job roles. Manually screening resumes is:

Time-consuming

Prone to bias

Inefficient at scale

This project builds an automated system to classify resumes into job categories using Machine Learning.

🧠 Solution Approach

The project follows a structured ML pipeline:

1️⃣ Data Collection

Resume dataset containing labeled job categories

2️⃣ Data Preprocessing

Removal of URLs

Removal of special characters

Lowercasing

Whitespace normalization

3️⃣ Feature Engineering

TF-IDF Vectorization

Maximum features limited for lightweight deployment

4️⃣ Model Training

Logistic Regression (chosen for deployment efficiency and performance)

Train-Test split

Accuracy evaluation

Confusion matrix analysis

5️⃣ Model Deployment

Saved trained model using Pickle

Built interactive Streamlit web application

Integrated PDF upload functionality

Displayed prediction confidence chart

🏗️ Project Architecture
Resume Text (PDF Upload)
        ↓
Text Extraction (pdfplumber)
        ↓
Text Cleaning (Regex Processing)
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression Model
        ↓
Predicted Category + Confidence Score

📊 Model Performance

Model Used: Logistic Regression

Vectorizer: TF-IDF (Max Features Limited)

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report

The model was optimized for both:

Performance

Lightweight deployment (<25MB)

🖥️ Application Features

✅ Drag & Drop PDF Resume Upload
✅ Automatic Text Extraction
✅ Resume Cleaning & Processing
✅ Job Category Prediction
✅ Top 5 Prediction Confidence Visualization
✅ Interactive Streamlit Interface
✅ Cloud Deployment

🛠️ Tech Stack
🔹 Programming

Python

🔹 Machine Learning

Scikit-learn

TF-IDF Vectorizer

Logistic Regression

🔹 NLP

Regex-based text preprocessing

🔹 Deployment

Streamlit

Streamlit Community Cloud

🔹 Libraries

pandas

numpy

scikit-learn

pdfplumber

pickle
