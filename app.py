import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# App Title
# ----------------------------
st.title("AI-Based Job Role Recommendation System")
st.write("Recommend suitable job roles based on skills and experience level")

# ----------------------------
# Dataset
# ----------------------------
data = {
    "Job_Role": [
        "Data Scientist",
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Analyst",
        "Software Developer"
    ],
    "Skills": [
        "python statistics machine learning pandas numpy",
        "python scikit-learn model training numpy",
        "python nlp tensorflow deep learning",
        "sql excel power bi data visualization",
        "java python data structures algorithms"
    ],
    "Experience_Level": [
        "Intermediate",
        "Intermediate",
        "Advanced",
        "Beginner",
        "Intermediate"
    ]
}

df = pd.DataFrame(data)

# ----------------------------
# User Inputs
# ----------------------------
user_skills = st.text_area(
    "Enter your skills (comma separated)",
    "python machine learning scikit-learn"
)

user_experience = st.selectbox(
    "Select your experience level",
    ["Beginner", "Intermediate", "Advanced"]
)

# ----------------------------
# Recommendation Logic
# ----------------------------
if st.button("Recommend Job Roles"):

    # NLP Vectorization
    vectorizer = TfidfVectorizer()
    job_tfidf = vectorizer.fit_transform(df["Skills"])
    user_tfidf = vectorizer.transform([user_skills])

    # Similarity
    similarity_scores = cosine_similarity(user_tfidf, job_tfidf)[0]

    # Experience bonus
    experience_bonus = [
        0.05 if level == user_experience else 0.0
        for level in df["Experience_Level"]
    ]

    # Final score
    df["Final_Score"] = similarity_scores + experience_bonus
    df["Match_Percentage"] = (
        df["Final_Score"] / df["Final_Score"].max()
    ) * 100

    # Top Recommendations
    top_jobs = df.sort_values(
        by="Match_Percentage", ascending=False
    ).head(3)

    st.subheader("Top Recommended Job Roles")
    for _, row in top_jobs.iterrows():
        st.write(
            f"**{row['Job_Role']}** — "
            f"{row['Match_Percentage']:.2f}% match"
        )
