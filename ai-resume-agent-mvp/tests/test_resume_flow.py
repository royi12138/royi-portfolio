from app.ai_analysis import calculate_scores, redact_sensitive_info, validate_analysis
from app.main import build_resume_json, parse_sections


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
