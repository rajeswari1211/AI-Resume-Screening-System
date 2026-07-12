import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening System")
st.caption("Developed by Rajeswari Kamepalli")
st.write("Compare resumes with job descriptions using AI")
st.markdown("""
### Project Description

This AI Resume Screening System uses Artificial Intelligence (AI) and Natural Language Processing (NLP) to compare a candidate's resume with a job description. It calculates a similarity score using CountVectorizer and Cosine Similarity, helping recruiters quickly identify suitable candidates and simplify the resume screening process.
""")

resume = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

if st.button("Calculate Match"):
    if resume and job_desc:
        text = [resume, job_desc]

        cv = CountVectorizer()
        matrix = cv.fit_transform(text)

        similarity = cosine_similarity(matrix)[0][1]
        score = round(similarity * 100, 2)

        st.subheader(f"Match Score: {score}%")

        if score > 70:
            st.success("Excellent Match")
        elif score > 40:
            st.warning("Average Match")
        else:
            st.error("Low Match")
    else:
        st.warning("Please enter both the resume text and the job description.")