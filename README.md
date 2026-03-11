<div align="center">

# 📄 AI Resume Analyzer

**An intelligent resume-to-job-description matching tool powered by Sentence Transformers and Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-orange?style=for-the-badge)](https://www.sbert.net/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🖼️ Demo

> _Screenshot placeholder — run the app locally and replace this section with a screenshot of the Streamlit UI._

```
Upload PDF → Paste Job Description → Analyze → View scores & download report
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📎 **PDF Resume Upload** | Extracts text from PDF resumes using `pdfplumber` |
| 🧠 **Semantic Similarity** | Compares resume & JD using `all-MiniLM-L6-v2` sentence embeddings |
| 🎯 **Skill Matching** | Detects technical skills and computes a keyword match score |
| 📊 **Visual Scores** | Three animated progress bars: Skill Match, Semantic Similarity, Overall |
| 🏅 **Performance Grade** | Excellent / Good / Needs Improvement — colour-coded badge |
| 💊 **Skill Pills** | Matched (green) and missing (red) skills at a glance |
| 🐍 **CLI Support** | Run `resume_analyzer.py` directly from the terminal with `.pdf` or `.txt` input |
| 🐳 **Docker Ready** | One-command startup with model cache persistence |
| ✅ **Unit Tested** | 60 pytest test cases with `--cov` coverage reporting |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Web UI** | Streamlit |
| **Semantic NLP** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **PDF Parsing** | `pdfplumber` |
| **ML / Math** | `scikit-learn` (cosine similarity utilities) |
| **Testing** | `pytest` + `pytest-cov` |
| **Containerisation** | Docker + Docker Compose |

---

## 📂 Project Structure

```
ai-resume-analyzer/
├── app.py                  # Streamlit frontend
├── resume_analyzer.py      # CLI analysis script
├── test_resume_analyzer.py # pytest test suite
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Compose with model cache volume
├── .dockerignore
├── sample_resume.txt       # Example resume (plain text)
└── job_description.txt     # Example job description
```

---

## 🚀 Installation & Usage

### Option 1 — Local (Python)

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/Shanky085/ai-resume-analyzer.git
cd ai-resume-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

### Option 2 — Docker 🐳

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows or Mac)

```bash
# Build image and start the container (first run downloads ~90 MB model)
docker compose up --build

# Run in background
docker compose up --build -d

# Stop the container
docker compose down

# Stop and remove the model cache volume
docker compose down -v
```

Open **http://localhost:8501** in your browser.

> **Tip:** The `all-MiniLM-L6-v2` model is cached in a named Docker volume (`hf_cache`). Subsequent starts are instant — no re-download.

---

## 🖥️ Usage Guide

### Streamlit App (Recommended)

1. **Upload Resume** — click the upload area and choose a `.pdf` file
2. **Paste Job Description** — paste the full JD into the text area
3. **Analyze Resume** — click the **🔍 Analyze Resume** button
4. **View Results:**
   - Three progress bars showing Skill Match %, Semantic Similarity %, Overall Match %
   - A colour-coded grade badge (🟢 Excellent / 🔵 Good / 🟡 Needs Improvement)
   - Matched skills (green pills) and missing skills (red pills)

### CLI Script

```bash
# Default: reads sample_resume.txt and job_description.txt
python resume_analyzer.py

# Pass custom files — supports .pdf and .txt
python resume_analyzer.py my_resume.pdf job_description.txt

# Both as PDF
python resume_analyzer.py resume.pdf job.pdf
```

Outputs a formatted report to the terminal and saves it as `analysis_report.txt`.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest test_resume_analyzer.py -v

# Run with coverage report
pytest test_resume_analyzer.py -v --cov=resume_analyzer --cov-report=term-missing
```

**Test coverage targets:** `preprocess`, `read_file`, skill extraction, cosine similarity math, weighted scoring formula, grade thresholds, edge cases (empty inputs, no overlap), and module-level output.

---

## 📊 How Scoring Works

```
Overall Match = 0.6 × Skill Match Score + 0.4 × Semantic Similarity Score
```

| Score Range | Grade |
|---|---|
| ≥ 80% | 🏅 Excellent Match |
| 60–79% | 🎯 Good Match |
| < 60% | ⚠️ Needs Improvement |

- **Skill Match Score** — percentage of job-required technical keywords found in the resume
- **Semantic Similarity** — cosine similarity between `all-MiniLM-L6-v2` sentence embeddings of resume and JD

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes, add tests, then push
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

**Guidelines:**
- Follow PEP 8 style
- Add/update tests in `test_resume_analyzer.py` for any changed logic
- Keep `resume_analyzer.py` CLI-compatible (no Streamlit imports)
- Run `pytest` before submitting

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and 🐍 · [Report a Bug](https://github.com/Shanky085/ai-resume-analyzer/issues) · [Request a Feature](https://github.com/Shanky085/ai-resume-analyzer/issues)

</div>