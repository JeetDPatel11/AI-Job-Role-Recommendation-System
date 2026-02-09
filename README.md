# AI-Job-Role-Recommendation-System


## Overview
This project is an AI-based job role recommendation system that suggests suitable job roles based on user skills and experience level. It uses Natural Language Processing (NLP) and Machine Learning techniques to compute similarity between user profiles and job role requirements.

## Features
- Skill-based job role recommendation
- NLP using TF-IDF vectorization
- Cosine similarity for matching
- Experience-level contextual weighting
- Deployed as an interactive web app using Streamlit

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLP (TF-IDF)
- Streamlit

## How It Works
1. User inputs skills and selects experience level
2. Skills are converted into TF-IDF vectors
3. Cosine similarity is calculated between user skills and job role requirements
4. Experience level is used to refine recommendations
5. Top job roles are displayed with match percentage

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

