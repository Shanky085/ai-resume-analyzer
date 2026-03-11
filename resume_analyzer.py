import re
import sys
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load .env (silently ignored if the file doesn't exist)
load_dotenv()

ST_MODEL_NAME    = os.getenv("ST_MODEL_NAME",    "all-MiniLM-L6-v2")
DEFAULT_RESUME   = os.getenv("DEFAULT_RESUME_PATH", "sample_resume.txt")
DEFAULT_JD       = os.getenv("DEFAULT_JD_PATH",     "job_description.txt")


def read_file(path: str) -> str:
    """Read text from a .pdf or .txt file."""
    if path.lower().endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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

# Read input files (supports .pdf and .txt)
_resume_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
_job_path    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JD
resume_text = read_file(_resume_path)
job_text    = read_file(_job_path)

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

# ---- Semantic Similarity (sentence-transformers) ----
_model = SentenceTransformer(ST_MODEL_NAME)
_embeddings = _model.encode([resume_text, job_text], convert_to_tensor=True)
semantic_percentage = float(util.cos_sim(_embeddings[0], _embeddings[1])[0][0]) * 100

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
import sys
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

ST_MODEL_NAME    = os.getenv("ST_MODEL_NAME",    "all-MiniLM-L6-v2")
DEFAULT_RESUME   = os.getenv("DEFAULT_RESUME_PATH", "sample_resume.txt")
DEFAULT_JD       = os.getenv("DEFAULT_JD_PATH",     "job_description.txt")


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

# Read input files (supports .pdf and .txt)
_resume_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
_job_path    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JD
resume_text = read_file(_resume_path)
job_text    = read_file(_job_path)

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

# ---- Semantic Similarity (sentence-transformers) ----
_model = SentenceTransformer(ST_MODEL_NAME)
_embeddings = _model.encode([resume_text, job_text], convert_to_tensor=True)
semantic_percentage = float(util.cos_sim(_embeddings[0], _embeddings[1])[0][0]) * 100

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
Semantic Similarity (sentence-transformers): {semantic_percentage:.2f}%

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