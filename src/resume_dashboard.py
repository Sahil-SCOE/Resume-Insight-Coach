from pathlib import Path
from collections import Counter
import re

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Resume Dataset Dashboard", layout="wide")
st.title("📊 Resume Dataset Insight Dashboard")
st.caption("Interactive visualization for the synthetic resume corpus.")

st.markdown(
    "### Open the Resume Coach on port 8501\n"
    "Use this dashboard on port 8502 to explore the dataset, then open the resume coach to upload a resume file and get AI-powered improvement suggestions."
)
st.link_button("Open AI Resume Coach", "http://localhost:8501", type="primary")


@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "resume_dataset.csv"
    df = pd.read_csv(data_path)
    df["resume_length"] = df["resume_text"].str.len()
    return df


@st.cache_data
def extract_keywords(df: pd.DataFrame, top_n: int = 15):
    stop_words = {
        "with", "years", "of", "experience", "education", "skills", "experience",
        "looking", "for", "opportunities", "as", "a", "to", "apply", "in", "the",
        "and", "or", "on", "was", "are", "their", "this", "that", "from"
    }

    all_words = []
    for text in df["resume_text"].astype(str):
        words = re.findall(r"[A-Za-z][A-Za-z.-]*", text.lower())
        all_words.extend(word for word in words if word not in stop_words and len(word) > 3)

    return Counter(all_words).most_common(top_n)


df = load_data()

categories = sorted(df["category"].unique())
selected_categories = st.multiselect(
    "Choose categories to view",
    options=categories,
    default=categories,
)
filtered_df = df[df["category"].isin(selected_categories)].copy()

st.subheader("Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total resumes", len(filtered_df))
col2.metric("Categories shown", filtered_df["category"].nunique())
col3.metric("Average resume length", f"{filtered_df['resume_length'].mean():.0f} chars")

st.subheader("Key Insights")
insight_text = [
    f"The dataset is perfectly balanced with {filtered_df['category'].value_counts().max()} resumes in each visible category.",
    f"The most common category in the filtered view is {filtered_df['category'].value_counts().idxmax()}.",
    f"The average resume length is about {filtered_df['resume_length'].mean():.0f} characters, which keeps the text samples compact and consistent.",
]
for item in insight_text:
    st.markdown(f"- {item}")

st.divider()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Category Distribution")
    dist = filtered_df["category"].value_counts().reset_index()
    dist.columns = ["category", "count"]
    fig_dist = px.bar(
        dist,
        x="count",
        y="category",
        orientation="h",
        color="category",
        title="Resume count by category",
        labels={"count": "Number of resumes", "category": "Category"},
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, width="stretch")

with right_col:
    st.subheader("Resume Length by Category")
    fig_box = px.box(
        filtered_df,
        x="category",
        y="resume_length",
        color="category",
        title="Resume length distribution",
        labels={"resume_length": "Characters", "category": "Category"},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

st.divider()

st.subheader("Most Common Technical Keywords")
keyword_data = extract_keywords(filtered_df, top_n=15)
keyword_df = pd.DataFrame(keyword_data, columns=["keyword", "count"])

fig_keywords = px.bar(
    keyword_df,
    x="count",
    y="keyword",
    orientation="h",
    color="count",
    title="Top resume keywords across the selected categories",
    labels={"keyword": "Keyword", "count": "Frequency"},
)
fig_keywords.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_keywords, width="stretch")

st.divider()

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), width="stretch")

st.divider()
st.subheader("🚀 AI Resume Help")
st.caption("First review the dataset insights on this page, then click below to open the resume coach and check your own resume.")
st.link_button("AI Resume Help → Open Resume Coach", "http://localhost:8501", type="primary")

st.markdown("### Insights")
st.markdown(
    "- The dataset is perfectly balanced, with 40 resumes per category across 10 categories.\n"
    "- The most frequent keywords reveal strong domain signals such as Python, Java, business, engineer, and marketing.\n"
    "- This is a synthetic dataset designed for classifier experiments, so the distribution is intentionally uniform."
)
