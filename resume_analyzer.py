import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined technical skills
tech_skills = {
    "python", "machine", "learning", "data", "analysis",
    "sql", "java", "c++", "flask", "django",
    "azure", "aws", "ai", "nlp"
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s+]', '', text)
    words = set(text.split())
    return words

# Read files
with open("sample_resume.txt", "r") as file:
    resume_text = file.read()

with open("job_description.txt", "r") as file:
    job_text = file.read()

# ---- Skill-Based Analysis ----
resume_words = preprocess(resume_text)
job_words = preprocess(job_text)

resume_skills = resume_words.intersection(tech_skills)
job_required_skills = job_words.intersection(tech_skills)

matched_skills = resume_skills.intersection(job_required_skills)
missing_skills = job_required_skills - resume_skills

if len(job_required_skills) != 0:
    skill_match_percentage = (len(matched_skills) / len(job_required_skills)) * 100
else:
    skill_match_percentage = 0

# ---- Semantic Similarity (TF-IDF) ----
documents = [resume_text, job_text]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
semantic_percentage = similarity_score[0][0] * 100

# ---- Performance Grade ----
final_score = (0.6 * skill_match_percentage) + (0.4 * semantic_percentage)

if final_score >= 80:
    grade = "Excellent Match"
elif final_score >= 60:
    grade = "Good Match"
else:
    grade = "Needs Improvement"

# ---- Generate Report ----
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined technical skills
tech_skills = {
    "python", "machine", "learning", "data", "analysis",
    "sql", "java", "c++", "flask", "django",
    "azure", "aws", "ai", "nlp"
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s+]', '', text)
    words = set(text.split())
    return words

# Read files
with open("sample_resume.txt", "r") as file:
    resume_text = file.read()

with open("job_description.txt", "r") as file:
    job_text = file.read()

# ---- Skill-Based Analysis ----
resume_words = preprocess(resume_text)
job_words = preprocess(job_text)

resume_skills = resume_words.intersection(tech_skills)
job_required_skills = job_words.intersection(tech_skills)

matched_skills = resume_skills.intersection(job_required_skills)
missing_skills = job_required_skills - resume_skills

if len(job_required_skills) != 0:
    skill_match_percentage = (len(matched_skills) / len(job_required_skills)) * 100
else:
    skill_match_percentage = 0

# ---- Semantic Similarity (TF-IDF) ----
documents = [resume_text, job_text]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
semantic_percentage = similarity_score[0][0] * 100

# ---- Performance Grade ----
final_score = (0.6 * skill_match_percentage) + (0.4 * semantic_percentage)

if final_score >= 80:
    grade = "Excellent Match"
elif final_score >= 60:
    grade = "Good Match"
else:
    grade = "Needs Improvement"

# ---- Generate Report ----
report = f"""
================ Resume Analysis Report ================

Skill-Based Match Percentage: {skill_match_percentage:.2f}%
Semantic Similarity (TF-IDF): {semantic_percentage:.2f}%

--------------------------------------------------------
Job Required Skills: {sorted(job_required_skills)}
Your Skills: {sorted(resume_skills)}

Matched Skills: {sorted(matched_skills)}
Missing Skills: {sorted(missing_skills)}

Overall Performance Grade: {grade}

========================================================
"""


print(report)

with open("analysis_report.txt", "w") as file:
    file.write(report)