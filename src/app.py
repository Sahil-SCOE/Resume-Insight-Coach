from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Resume Analyzer & Job Matcher", layout="wide")
st.title("📄 Intelligent Resume Analyzer & Job Matcher")
st.markdown("**Built by Sahil Shaikh**")


CATEGORY_SKILLS = {
    "Data Scientist": {
        "add": ["Python", "SQL", "statistics", "scikit-learn", "Tableau"],
        "remove": ["generic buzzwords", "long unrelated experience lists"],
    },
    "Web Developer": {
        "add": ["HTML", "CSS", "JavaScript", "React", "REST APIs"],
        "remove": ["unclear UI claims", "irrelevant backend jargon"],
    },
    "Software Engineer": {
        "add": ["Data Structures", "Algorithms", "Git", "unit testing", "Docker"],
        "remove": ["vague responsibility statements", "non-technical claims"],
    },
    "HR": {
        "add": ["recruitment", "onboarding", "employee engagement", "HRIS"],
        "remove": ["unrelated engineering terms", "unclear salary details"],
    },
    "Business Analyst": {
        "add": ["Excel", "SQL", "requirements gathering", "stakeholder management"],
        "remove": ["unsupported metrics", "mixed role descriptions"],
    },
    "Mechanical Engineer": {
        "add": ["AutoCAD", "SolidWorks", "CAD design", "quality control"],
        "remove": ["software-only skills", "irrelevant sales terminology"],
    },
    "Sales": {
        "add": ["CRM", "lead generation", "negotiation", "Salesforce"],
        "remove": ["overly technical jargon", "weak target numbers"],
    },
    "Marketing": {
        "add": ["SEO", "Google Analytics", "content marketing", "social media marketing"],
        "remove": ["unsupported campaign claims", "unrelated technical tools"],
    },
    "Java Developer": {
        "add": ["Java", "Spring Boot", "REST APIs", "Hibernate", "JUnit"],
        "remove": ["vague architecture statements", "mixed-language claims"],
    },
    "DevOps Engineer": {
        "add": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform"],
        "remove": ["unrelated compliance wording", "unclear system claims"],
    },
}


def get_local_suggestions(predicted_category):
    data = CATEGORY_SKILLS.get(predicted_category, {
        "add": ["relevant domain skills"],
        "remove": ["generic buzzwords"],
    })
    return data


@st.cache_resource
def load_model_artifacts():
    model_path = Path(__file__).resolve().parents[1] / "models" / "resume_classifier.joblib"
    vectorizer_path = Path(__file__).resolve().parents[1] / "models" / "tfidf_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


model, vectorizer = load_model_artifacts()

st.subheader("📄 Resume Input")
uploaded_file = st.file_uploader(
    "Upload a resume file (TXT/MD/CSV)",
    type=["txt", "md", "csv"],
    help="Upload a resume file or paste text below. The uploaded content will replace the text area automatically.",
)

resume = ""
if uploaded_file is not None:
    try:
        resume = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        resume = str(uploaded_file.read())

resume = st.text_area(
    "Paste or replace your resume text here",
    value=resume,
    height=250,
    help="Edit the text here before clicking AI Help.",
)

st.subheader("🔎 Resume Learning Flow")
info_col, api_col = st.columns([2, 1])
with info_col:
    st.info("The model predicts the job category and highlights the most important resume keywords. If you provide an AI API key, the app can suggest skills to add or remove to improve the resume.")
with api_col:
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

st.markdown("### AI Help")
st.caption("Upload a resume, edit the text, then click the highlighted button below to analyze it.")
if st.button("AI Help & Analyze Resume", key="predict_btn", type="primary"):
    if not resume.strip():
        st.warning("Paste a resume above first.")
    else:
        X = vectorizer.transform([resume])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        st.subheader(f"Predicted Category: **{pred}**")

        probs_df = pd.DataFrame({
            "category": model.classes_,
            "confidence": proba,
        }).sort_values("confidence", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(probs_df["category"], probs_df["confidence"], color="#4C72B0")
        ax.set_xlabel("Confidence")
        ax.set_title("Category Prediction Confidence")
        st.pyplot(fig)

        feature_names = vectorizer.get_feature_names_out()
        top_idx = X.toarray()[0].argsort()[-12:][::-1]
        top_terms = [(feature_names[i], round(X.toarray()[0][i], 3)) for i in top_idx if X.toarray()[0][i] > 0]

        if top_terms:
            st.subheader("Top Keywords Driving This Prediction")
            terms_df = pd.DataFrame(top_terms, columns=["term", "tf-idf weight"])
            st.dataframe(terms_df, width="stretch")

        suggestions = get_local_suggestions(pred)
        st.subheader("💡 Suggested Skills to Add / Remove")
        add_list = ", ".join(suggestions["add"])
        remove_list = ", ".join(suggestions["remove"])
        st.markdown(f"**Add:** {add_list}")
        st.markdown(f"**Remove or reduce:** {remove_list}")

        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                prompt = f"""
You are an expert resume coach.
Given this resume text and the predicted job category '{pred}', suggest:
1. Skills to add to make the resume stronger for that category.
2. Skills or phrases to remove or reduce if they are weak or unrelated.
3. Short improvement actions.
Return the answer in simple bullet points.

Resume:
{resume}
"""
                with st.spinner("Generating AI-powered resume improvement suggestions..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    st.subheader("🤖 AI Resume Improvement Suggestions")
                    st.write(response.choices[0].message.content)
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}")
        else:
            st.info("Add your OpenAI API key to get AI-based skill add/remove recommendations. The local suggestions above still help you understand the improvement direction.")

st.markdown("---")
st.markdown("### AI Help Button")
st.caption("Use this resume coach page to upload a resume, edit the text, and click the prediction button to get a strong category + skill suggestion flow.")

st.divider()

st.subheader("📊 Training Data Snapshot")
training_image = Path(__file__).resolve().parents[1] / "outputs" / "category_distribution.png"
if training_image.exists():
    st.image(str(training_image), caption="Training data category distribution")
else:
    st.info("Run the training script once to generate the visualization assets.")
