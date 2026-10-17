from app.ai_analysis import calculate_scores, redact_sensitive_info, validate_analysis
from app.main import app, build_resume_json, parse_sections
from fastapi.testclient import TestClient

client = TestClient(app)

SAMPLE_RESUME = "拥有多年互联网产品管理经验，负责需求规划、跨部门协作和产品交付。持续跟进用户反馈，推动项目按计划落地。"


def test_scores_are_explainable_and_bounded():
    score = calculate_scores("教育经历\n本科\n工作经历\n负责上线并提升用户数 20%", [])
    assert 0 <= score["overall"] <= 100
    assert len(score["dimensions"]) == 4
    assert all("reason" in item and item["weight"] > 0 for item in score["dimensions"])


def test_analysis_only_keeps_exact_source_quotes():
    source = "负责产品上线并提升用户数"
    result = validate_analysis(
        {"summary": "ok", "issues": [
            {"source_quote": source, "problem": "可补充结果"},
            {"source_quote": "没有出现在原文中的内容", "problem": "无依据"},
        ]},
        source,
    )
    assert len(result["issues"]) == 1
    assert result["issues"][0]["source_quote"] == source
    assert "scores" in result


def test_common_identifiers_are_redacted():
    redacted = redact_sensitive_info("邮箱 a.user@example.com，电话 13800138000")
    assert "a.user@example.com" not in redacted
    assert "13800138000" not in redacted


def test_resume_parser_keeps_sections_and_raw_text():
    raw = "个人信息\n张三\n教育经历\n本科\n工作经历\n负责产品上线\n技能\nPython"
    sections = parse_sections(raw)
    assert sections["education"] == ["本科"]
    assert sections["work_experience"] == ["负责产品上线"]
    resume = build_resume_json("resume.docx", ".docx", raw, 1)
    assert resume["raw_text"] == raw


def test_analysis_requires_privacy_consent():
    response = client.post("/api/resume/analyze", json={"resume_text": SAMPLE_RESUME, "privacy_consent": False})
    assert response.status_code == 400


def test_revision_requires_privacy_consent():
    response = client.post("/api/resume/revise", json={
        "resume_text": SAMPLE_RESUME,
        "accepted_issues": [{"source_quote": "拥有多年互联网产品管理经验", "suggestion": "优化表达"}],
        "privacy_consent": False,
    })
    assert response.status_code == 400


def test_jd_match_requires_privacy_consent():
    response = client.post("/api/resume/match", json={
        "resume_text": SAMPLE_RESUME,
        "jd_text": "招聘高级产品经理，负责产品规划、用户调研、需求分析、跨团队沟通与项目交付。",
        "privacy_consent": False,
    })
    assert response.status_code == 400


def test_docx_export_returns_word_document():
    response = client.post("/api/resume/export", json={"resume_text": SAMPLE_RESUME, "filename": "测试简历"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert response.content.startswith(b"PK")
