import streamlit as st
from openai import OpenAI

st.title("📄 Intelligent Resume Analyzer & Job Matcher")
st.markdown("**Built by Sahil Shaikh**")

api_key = st.text_input("Enter OpenAI API Key", type="password")

if api_key:
    client = OpenAI(api_key=api_key)

    resume = st.text_area("Paste your Resume Text here", height=300)
    
    if st.button("Analyze Resume & Suggest Jobs"):
        with st.spinner("Analyzing your resume..."):
            prompt = f"""
            Analyze this resume and suggest 5 suitable job roles with reasons.
            Also suggest skills to improve and companies to target.

            Resume:
            {resume}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            st.subheader("📋 Analysis Result")
            st.write(response.choices[0].message.content)