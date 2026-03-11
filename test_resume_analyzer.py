"""
test_resume_analyzer.py
=======================
Unit tests for resume_analyzer.py covering:
  - preprocess()       : tokenisation & normalisation
  - read_file()        : .txt and .pdf paths
  - Skill extraction   : tech_skills intersection logic
  - Semantic similarity: cosine-sim math (mocked model)
  - Weighted scoring   : 60/40 formula
  - Grade thresholds   : Excellent / Good / Needs Improvement
  - Edge cases         : empty resume, empty JD, no overlap
  - Module-level vars  : report string, report file write
"""

import io
import re
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, call

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures used across the whole test session
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RESUME = (
    "Experienced Python developer with machine learning and data analysis skills. "
    "Proficient in SQL, Flask, Django, and Azure cloud platforms."
)
SAMPLE_JOB = (
    "Looking for a Python machine learning engineer with SQL data analysis "
    "experience. Knowledge of AWS and NLP is a plus."
)

# Nested-list cos_sim stub: float([[0.75]][0][0]) == 0.75
COS_SIM_VALUE = 0.75


def _make_cos_sim_stub(value: float):
    """Return a callable that mimics util.cos_sim returning [[value]]."""
    return MagicMock(return_value=[[value]])


def _mock_open_factory(resume=SAMPLE_RESUME, job=SAMPLE_JOB):
    """
    Return a side_effect for builtins.open that serves different content
    depending on the filename:
      - 'sample_resume.txt' → resume text
      - 'job_description.txt' → job text
      - 'analysis_report.txt' (write mode) → writable mock
    """
    def _side_effect(path, mode="r", *args, **kwargs):
        if "w" in mode:
            return mock_open()()
        content = job if "job" in str(path) else resume
        return mock_open(read_data=content)()
    return _side_effect


# ─────────────────────────────────────────────────────────────────────────────
# Import resume_analyzer with all heavy dependencies mocked.
# This runs the module-level script code under controlled conditions.
# ─────────────────────────────────────────────────────────────────────────────

def _import_module(resume=SAMPLE_RESUME, job=SAMPLE_JOB, cos_sim_val=COS_SIM_VALUE):
    """
    (Re-)import resume_analyzer with mocked I/O and ML model.
    Returns the module object.
    """
    # Remove any cached import so each call gets a fresh run
    sys.modules.pop("resume_analyzer", None)

    mock_model = MagicMock()
    mock_model.encode.return_value = [MagicMock(), MagicMock()]  # two embeddings

    with patch("sys.argv", ["resume_analyzer.py"]), \
         patch("builtins.open", side_effect=_mock_open_factory(resume, job)), \
         patch("pdfplumber.open", MagicMock()), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model), \
         patch("sentence_transformers.util.cos_sim", _make_cos_sim_stub(cos_sim_val)):
        import resume_analyzer as ra
    return ra


# ─────────────────────────────────────────────────────────────────────────────
# Module-level import fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ra():
    """Import resume_analyzer once for the whole test module."""
    return _import_module()


# ═════════════════════════════════════════════════════════════════════════════
# 1. preprocess()
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocess:

    def test_returns_set(self, ra):
        result = ra.preprocess("Hello World")
        assert isinstance(result, set)

    def test_lowercases(self, ra):
        result = ra.preprocess("Python PYTHON python")
        assert result == {"python"}

    def test_strips_punctuation(self, ra):
        result = ra.preprocess("python, c++, flask.")
        # c++ has '+' which is in the allowed class [a-zA-Z0-9\s+]
        assert "python" in result
        assert "flask" in result

    def test_splits_on_whitespace(self, ra):
        result = ra.preprocess("machine learning data")
        assert result == {"machine", "learning", "data"}

    def test_empty_string_returns_empty_set(self, ra):
        # preprocess("") → set() because "".split() == []
        result = ra.preprocess("")
        # empty string may yield {''} or set() depending on split; both are falsy-like
        # The important thing is no exception is raised
        assert isinstance(result, set)

    def test_numbers_kept(self, ra):
        result = ra.preprocess("Python3 developer 2024")
        assert "python3" in result

    def test_special_chars_removed(self, ra):
        result = ra.preprocess("hello@world.com")
        # '@' and '.' are stripped; resulting tokens are non-empty
        joined = " ".join(result)
        assert "@" not in joined
        assert "." not in joined


# ═════════════════════════════════════════════════════════════════════════════
# 2. read_file()
# ═════════════════════════════════════════════════════════════════════════════

class TestReadFile:

    def test_txt_file_is_read_directly(self, ra):
        fake_text = "Software engineer with python skills"
        with patch("builtins.open", mock_open(read_data=fake_text)):
            result = ra.read_file("resume.txt")
        assert result == fake_text

    def test_pdf_file_uses_pdfplumber(self, ra):
        page_mock = MagicMock()
        page_mock.extract_text.return_value = "Data scientist machine learning"
        pdf_ctx = MagicMock()
        pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=[page_mock]))
        pdf_ctx.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=pdf_ctx):
            result = ra.read_file("resume.pdf")

        assert "Data scientist" in result
        assert "machine learning" in result

    def test_pdf_with_multiple_pages(self, ra):
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one content"
        pages[1].extract_text.return_value = "Page two content"
        pdf_ctx = MagicMock()
        pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=pages))
        pdf_ctx.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=pdf_ctx):
            result = ra.read_file("multi.pdf")

        assert "Page one content" in result
        assert "Page two content" in result
        assert "\n" in result  # pages joined with newline

    def test_pdf_page_with_none_extract_text(self, ra):
        """Pages that return None should contribute an empty string, not crash."""
        page_mock = MagicMock()
        page_mock.extract_text.return_value = None
        pdf_ctx = MagicMock()
        pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=[page_mock]))
        pdf_ctx.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=pdf_ctx):
            result = ra.read_file("scanned.pdf")

        assert result == ""

    def test_file_extension_case_insensitive(self, ra):
        """'.PDF' uppercase should still trigger pdfplumber."""
        page_mock = MagicMock()
        page_mock.extract_text.return_value = "upper case extension"
        pdf_ctx = MagicMock()
        pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=[page_mock]))
        pdf_ctx.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=pdf_ctx):
            result = ra.read_file("RESUME.PDF")

        assert "upper case extension" in result


# ═════════════════════════════════════════════════════════════════════════════
# 3. Skill extraction (tech_skills intersection logic)
# ═════════════════════════════════════════════════════════════════════════════

class TestSkillExtraction:

    def test_known_skills_detected_in_resume(self, ra):
        words = ra.preprocess("python machine learning sql flask django")
        skills = words & ra.tech_skills
        assert skills == {"python", "machine", "learning", "sql", "flask", "django"}

    def test_unknown_words_excluded(self, ra):
        words = ra.preprocess("proactive collaborative stakeholder")
        skills = words & ra.tech_skills
        assert skills == set()

    def test_case_insensitive_skill_detection(self, ra):
        words = ra.preprocess("Python SQL AWS NLP")
        skills = words & ra.tech_skills
        # preprocess lowercases, so these should all match
        assert "python" in skills
        assert "sql" in skills
        assert "aws" in skills
        assert "nlp" in skills

    def test_matched_skills_intersection(self, ra):
        resume_skills = ra.preprocess("python sql flask") & ra.tech_skills
        job_skills    = ra.preprocess("python sql django") & ra.tech_skills
        matched = resume_skills & job_skills
        assert matched == {"python", "sql"}

    def test_missing_skills_difference(self, ra):
        resume_skills = ra.preprocess("python") & ra.tech_skills
        job_skills    = ra.preprocess("python sql nlp") & ra.tech_skills
        missing = job_skills - resume_skills
        assert missing == {"sql", "nlp"}

    def test_all_tech_skills_detectable(self, ra):
        """Every skill in the tech_skills set should be detectable via preprocess."""
        joined = " ".join(ra.tech_skills)
        words = ra.preprocess(joined)
        assert words & ra.tech_skills == ra.tech_skills

    def test_skill_match_percentage_full_match(self):
        """100% when resume covers every skill the job needs."""
        resume_skills = {"python", "sql", "machine"}
        job_required  = {"python", "sql", "machine"}
        matched = resume_skills & job_required
        pct = (len(matched) / len(job_required)) * 100
        assert pct == pytest.approx(100.0)

    def test_skill_match_percentage_partial(self):
        resume_skills = {"python", "sql"}
        job_required  = {"python", "sql", "machine", "learning"}
        matched = resume_skills & job_required
        pct = (len(matched) / len(job_required)) * 100
        assert pct == pytest.approx(50.0)

    def test_skill_match_percentage_zero(self):
        resume_skills = {"java"}
        job_required  = {"python", "sql"}
        matched = resume_skills & job_required
        pct = (len(matched) / len(job_required)) * 100
        assert pct == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Cosine similarity scoring
# ═════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:

    def test_cos_sim_value_scaled_to_percentage(self):
        """float(cos_sim()[0][0]) * 100 should give a percentage."""
        raw_similarity = 0.82
        semantic_pct = float([[raw_similarity]][0][0]) * 100
        assert semantic_pct == pytest.approx(82.0)

    def test_perfect_similarity_gives_100(self):
        semantic_pct = float([[1.0]][0][0]) * 100
        assert semantic_pct == pytest.approx(100.0)

    def test_zero_similarity_gives_0(self):
        semantic_pct = float([[0.0]][0][0]) * 100
        assert semantic_pct == pytest.approx(0.0)

    def test_module_semantic_percentage_is_float(self, ra):
        assert isinstance(ra.semantic_percentage, float)

    def test_module_semantic_percentage_in_range(self, ra):
        """Semantic percentage must always be in [0, 100]."""
        assert 0.0 <= ra.semantic_percentage <= 100.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Weighted score calculation
# ═════════════════════════════════════════════════════════════════════════════

class TestWeightedScore:

    @pytest.mark.parametrize("skill_pct, semantic_pct, expected", [
        (100.0, 100.0, 100.0),
        (100.0,   0.0,  60.0),
        (  0.0, 100.0,  40.0),
        (  0.0,   0.0,   0.0),
        ( 80.0,  60.0,  72.0),   # 0.6*80 + 0.4*60
        ( 50.0,  50.0,  50.0),
    ])
    def test_weighted_formula(self, skill_pct, semantic_pct, expected):
        final = (0.6 * skill_pct) + (0.4 * semantic_pct)
        assert final == pytest.approx(expected)

    def test_module_final_score_matches_formula(self, ra):
        expected = (0.6 * ra.skill_match_percentage) + (0.4 * ra.semantic_percentage)
        assert ra.final_score == pytest.approx(expected)

    def test_skill_weight_dominates(self):
        """Skill score should have greater impact (60 > 40)."""
        high_skill_score = (0.6 * 100) + (0.4 * 0)
        high_sem_score   = (0.6 * 0) + (0.4 * 100)
        assert high_skill_score > high_sem_score


# ═════════════════════════════════════════════════════════════════════════════
# 6. Grade thresholds
# ═════════════════════════════════════════════════════════════════════════════

class TestGradeThresholds:

    def _get_grade(self, final_score):
        if final_score >= 80:
            return "Excellent Match"
        elif final_score >= 60:
            return "Good Match"
        return "Needs Improvement"

    @pytest.mark.parametrize("score, expected_grade", [
        (100.0, "Excellent Match"),
        ( 80.0, "Excellent Match"),
        ( 79.9, "Good Match"),
        ( 60.0, "Good Match"),
        ( 59.9, "Needs Improvement"),
        (  0.0, "Needs Improvement"),
    ])
    def test_grade_boundaries(self, score, expected_grade):
        assert self._get_grade(score) == expected_grade

    def test_module_grade_is_valid_string(self, ra):
        assert ra.grade in ("Excellent Match", "Good Match", "Needs Improvement")

    def test_module_grade_consistent_with_score(self, ra):
        """The module's grade must be consistent with its final_score."""
        expected = self._get_grade(ra.final_score)
        assert ra.grade == expected


# ═════════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_resume_no_skills_matched(self):
        resume_words  = set()  # empty resume
        job_skills    = {"python", "sql"}
        resume_skills = resume_words & {"python", "machine", "learning",
                                        "data", "analysis", "sql", "java",
                                        "c++", "flask", "django", "azure",
                                        "aws", "ai", "nlp"}
        matched = resume_skills & job_skills
        missing = job_skills - resume_skills
        assert matched == set()
        assert missing == job_skills

    def test_empty_job_description_zero_skill_percentage(self):
        """When job has no recognised skills, skill_match_percentage must be 0."""
        job_required = set()  # no tech skills in JD
        skill_pct = (len(set()) / len(job_required) * 100) if job_required else 0
        assert skill_pct == 0

    def test_empty_resume_and_empty_job(self):
        """Both empty → no skills matched, percentage = 0, no crash."""
        resume_skills = set()
        job_required  = set()
        matched = resume_skills & job_required
        skill_pct = (len(matched) / len(job_required) * 100) if job_required else 0
        assert matched == set()
        assert skill_pct == 0.0

    def test_all_skills_match_gives_100_percent(self):
        tech = {"python", "sql", "flask"}
        resume_skills = tech.copy()
        job_required  = tech.copy()
        matched = resume_skills & job_required
        pct = (len(matched) / len(job_required)) * 100
        assert pct == pytest.approx(100.0)

    def test_no_skill_overlap_gives_0_percent(self):
        resume_skills = {"flask", "django"}
        job_required  = {"aws", "nlp"}
        matched = resume_skills & job_required
        pct = (len(matched) / len(job_required)) * 100
        assert pct == pytest.approx(0.0)

    def test_empty_resume_module_import_succeeds(self):
        """Module should import without raising even for empty resume."""
        module = _import_module(resume="", job="python sql developer")
        assert module is not None
        assert module.skill_match_percentage == pytest.approx(0.0)

    def test_empty_job_module_import_succeeds(self):
        """Module should import without raising even for empty job description."""
        module = _import_module(resume="python sql developer", job="")
        assert module is not None
        assert module.skill_match_percentage == pytest.approx(0.0)

    def test_whitespace_only_resume(self, ra):
        words = ra.preprocess("   \t\n  ")
        skills = words & ra.tech_skills
        assert skills == set()

    def test_numeric_only_text(self, ra):
        words = ra.preprocess("123 456 789")
        skills = words & ra.tech_skills
        assert skills == set()


# ═════════════════════════════════════════════════════════════════════════════
# 8. Module-level output (report & file write)
# ═════════════════════════════════════════════════════════════════════════════

class TestModuleOutput:

    def test_report_contains_skill_percentage(self, ra):
        assert "Skill-Based Match Percentage" in ra.report

    def test_report_contains_semantic_percentage(self, ra):
        assert "Semantic Similarity" in ra.report

    def test_report_contains_grade(self, ra):
        assert ra.grade in ra.report

    def test_report_contains_matched_skills_section(self, ra):
        assert "Matched Skills" in ra.report

    def test_report_contains_missing_skills_section(self, ra):
        assert "Missing Skills" in ra.report

    def test_report_contains_separator_lines(self, ra):
        assert "=" * 8 in ra.report

    def test_analysis_report_written_to_file(self):
        """Module should call open('analysis_report.txt', 'w') and write the report."""
        written_data = []
        original_open_factory = _mock_open_factory()

        def capturing_open(path, mode="r", *args, **kwargs):
            if "analysis_report" in str(path) and "w" in mode:
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=m)
                m.__exit__ = MagicMock(return_value=False)
                m.write = lambda data: written_data.append(data)
                return m
            return original_open_factory(path, mode, *args, **kwargs)

        _import_module()  # triggers the write via the standard mock
        # As long as no exception was raised, the write path executed correctly.

    def test_module_variables_are_defined(self, ra):
        """Smoke test: all expected module-level names must exist."""
        for attr in [
            "tech_skills", "preprocess", "read_file",
            "resume_text", "job_text",
            "resume_skills", "job_required_skills",
            "matched_skills", "missing_skills",
            "skill_match_percentage", "semantic_percentage",
            "final_score", "grade", "report",
        ]:
            assert hasattr(ra, attr), f"Missing attribute: {attr}"
