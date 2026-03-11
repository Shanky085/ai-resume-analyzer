import re
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
}

/* Score row */
.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.score-label {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
}
.score-value {
    color: #a78bfa;
    font-size: 1rem;
    font-weight: 700;
}

/* Grade badge */
.grade-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 0.5rem;
}
.grade-excellent { background: linear-gradient(90deg,#10b981,#34d399); color:#fff; }
.grade-good      { background: linear-gradient(90deg,#3b82f6,#60a5fa); color:#fff; }
.grade-needs     { background: linear-gradient(90deg,#f59e0b,#fbbf24); color:#fff; }

/* Skill pills */
.pill-wrap { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.pill {
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
}
.pill-match   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.pill-missing { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* Section header */
.section-header {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Streamlit widget overrides */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stFileUploader {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(167,139,250,0.5) !important;
    border-radius: 12px !important;
}
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.4px;
    cursor: pointer;
    transition: opacity 0.2s ease;
}
div.stButton > button:hover { opacity: 0.88; }

/* Progress bar colour tweak */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#7c3aed,#60a5fa) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Analysis logic (mirrors resume_analyzer.py — file unchanged)
# ──────────────────────────────────────────────────────────────────────────────
TECH_SKILLS = {
    "python", "machine", "learning", "data", "analysis",
    "sql", "java", "c++", "flask", "django",
    "azure", "aws", "ai", "nlp"
}

@st.cache_resource(show_spinner="Loading semantic model…")
def _load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def preprocess(text: str) -> set:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s+]', '', text)
    return set(text.split())

def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def analyze(resume_text: str, job_text: str) -> dict:
    # Skill-based analysis
    resume_words       = preprocess(resume_text)
    job_words          = preprocess(job_text)
    resume_skills      = resume_words & TECH_SKILLS
    job_required       = job_words & TECH_SKILLS
    matched            = resume_skills & job_required
    missing            = job_required - resume_skills

    skill_pct = (len(matched) / len(job_required) * 100) if job_required else 0

    # Semantic similarity (sentence-transformers)
    model      = _load_model()
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    semantic_pct = float(util.cos_sim(embeddings[0], embeddings[1])[0][0]) * 100

    # Overall weighted score
    overall = 0.6 * skill_pct + 0.4 * semantic_pct

    if overall >= 80:
        grade = ("Excellent Match", "grade-excellent")
    elif overall >= 60:
        grade = ("Good Match", "grade-good")
    else:
        grade = ("Needs Improvement", "grade-needs")

    return {
        "skill_pct":    skill_pct,
        "semantic_pct": semantic_pct,
        "overall":      overall,
        "grade":        grade,
        "matched":      sorted(matched),
        "missing":      sorted(missing),
        "job_required": sorted(job_required),
        "resume_skills": sorted(resume_skills),
    }

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Analyzer</h1>
    <p>Upload your resume and paste a job description to see how well you match.</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="section-header">📎 Upload Resume (PDF)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload Resume",
        type=["pdf"],
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<p class="section-header">📝 Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        label="Job Description",
        placeholder="Paste the job description here…",
        height=180,
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Resume")

# ── Results ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF resume first.")
    elif not job_description.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        with st.spinner("Analyzing your resume…"):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please ensure it is not scanned/image-only.")
                st.stop()
            results = analyze(resume_text, job_description)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.5rem 0'>", unsafe_allow_html=True)

        # ── Score cards ───────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📊 Match Scores</p>', unsafe_allow_html=True)

        scores = [
            ("Skill Match Score",        results["skill_pct"],    "Based on matching technical skills from the job description."),
            ("Semantic Similarity Score", results["semantic_pct"], "Sentence-transformer (all-MiniLM-L6-v2) cosine similarity."),
            ("Overall Match",             results["overall"],      "Weighted score: 60% skill match + 40% semantic similarity."),
        ]

        for label, value, help_text in scores:
            pct = round(value, 1)
            st.markdown(f"""
            <div class="glass-card">
                <div class="score-row">
                    <span class="score-label">{label}</span>
                    <span class="score-value">{pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Clamp to [0,1] for st.progress
            st.progress(min(max(value / 100, 0.0), 1.0), text=help_text)

        # ── Grade badge ───────────────────────────────────────────────────────
        grade_text, grade_cls = results["grade"]
        st.markdown(f"""
        <div style="text-align:center;margin:1.5rem 0 0.5rem">
            <span class="grade-badge {grade_cls}">🏅 {grade_text}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.5rem 0'>", unsafe_allow_html=True)

        # ── Skills breakdown ──────────────────────────────────────────────────
        skills_col1, skills_col2 = st.columns(2, gap="large")

        with skills_col1:
            st.markdown('<p class="section-header">✅ Matched Skills</p>', unsafe_allow_html=True)
            if results["matched"]:
                pills = "".join(
                    f'<span class="pill pill-match">{s}</span>'
                    for s in results["matched"]
                )
                st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#64748b;font-size:0.9rem">No matching skills found.</span>',
                            unsafe_allow_html=True)

        with skills_col2:
            st.markdown('<p class="section-header">❌ Missing Skills</p>', unsafe_allow_html=True)
            if results["missing"]:
                pills = "".join(
                    f'<span class="pill pill-missing">{s}</span>'
                    for s in results["missing"]
                )
                st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#34d399;font-size:0.9rem">🎉 All required skills matched!</span>',
                            unsafe_allow_html=True)

        # ── Summary expander ──────────────────────────────────────────────────
        with st.expander("📋 Full Skills Summary"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Job Required Skills**")
                st.write(results["job_required"] or ["None detected"])
            with c2:
                st.markdown("**Your Skills (detected)**")
                st.write(results["resume_skills"] or ["None detected"])
