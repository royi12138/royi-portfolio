from app.ai_analysis import (build_match_messages, build_messages, build_revision_messages, build_completion_payload, calculate_scores, extract_completion_text, parse_json_object, redact_sensitive_info, validate_analysis, validate_resume_sections)
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


def test_homepage_prevents_stale_consent_script():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    assert "privacy_consent:document.getElementById('consent').checked" in response.text


def test_jd_image_upload_extracts_text(monkeypatch):
    jd = "招聘产品经理，负责产品规划、需求分析、用户研究、跨团队协作、项目推进与产品上线交付。"
    monkeypatch.setattr("app.main.extract_image", lambda path: (jd, 1))
    response = client.post(
        "/api/jd/parse",
        files={"file": ("岗位.png", b"image bytes", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["jd_text"] == jd


def test_jd_text_upload_is_rejected():
    response = client.post(
        "/api/jd/parse",
        files={"file": ("岗位.txt", b"job description long enough to pass", "text/plain")},
    )
    assert response.status_code == 400

def test_analysis_page_reveals_results_and_contains_comparison_views():
    response = client.get("/")
    assert response.status_code == 200
    assert "analysisResult').classList.remove('hidden')" in response.text
    assert 'id="beforeResume"' in response.text
    assert 'id="jdBeforeText"' in response.text
    assert "async function generateJDRevision()" in response.text
    assert "function renderComparison(beforeId,afterId,outputId)" in response.text
    assert "diff-change" in response.text
    assert 'accept=".jpg,.jpeg,.png,image/jpeg,image/png"' in response.text


def test_health_version_is_current():
    response = client.get("/api/health")
    assert response.json()["version"] == "0.6.0"



def test_completion_handles_text_blocks_and_glm_payload():
    response = {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}, {"content": " done"}]}}]}
    assert extract_completion_text(response) == "ok\n done"
    payload = build_completion_payload("glm-5.3-flash", [], json_mode=True)
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert payload["clear_thinking"] is True
    assert payload["response_format"] == {"type": "json_object"}


def test_match_json_parser_accepts_fenced_or_prefixed_json():
    assert parse_json_object("```json\n{\"summary\":\"ok\"}\n```") == {"summary": "ok"}
    assert parse_json_object("回答：{\"summary\":\"ok\"}完成") == {"summary": "ok"}


def test_structured_resume_summary_requires_exact_source_and_supported_numbers():
    source = "工作业绩：注册用户增长2000+，转化率提升12%。"
    result = validate_resume_sections(
        {
            "work_experience": [
                {"summary": "• 注册用户增长 2000+；转化率提升12%。", "source_quote": source},
                {"summary": "新增营收999万元。", "source_quote": source},
                {"summary": "完全无关的总结", "source_quote": "不存在于原文"},
            ]
        },
        source,
    )
    assert result["work_experience"] == ["注册用户增长 2000+；转化率提升12%。"]
    assert result["education"] == []


def test_analysis_returns_clean_structured_resume_sections():
    source = "教育经历：本科，信息管理专业。"
    result = validate_analysis({
        "summary": "清晰",
        "resume_sections": {
            "education": [{"summary": "• 本科，信息管理专业。", "source_quote": source}]
        },
        "issues": [],
    }, source)
    assert result["resume_sections"]["education"] == ["本科，信息管理专业。"]

def test_advisor_prompts_require_specific_evidence_and_usable_rewrites():
    analysis_messages = build_messages("负责产品规划与上线，推动用户反馈闭环。")
    analysis_prompt = analysis_messages[0]["content"]
    assert "可直接替换原文" in analysis_prompt
    assert "按影响从高到低排序" in analysis_prompt
    assert "不要为了“有问题可提”而挑小毛病" in analysis_prompt

    revision_messages = build_revision_messages(
        "工作经历\n负责产品规划与上线",
        [{"section": "工作经历", "source_quote": "负责产品规划与上线", "suggestion": "突出行动和结果"}],
        "用户确认上线范围为3个城市",
        "招聘产品经理，负责产品规划",
    )
    revision_system, revision_user = revision_messages[0]["content"], revision_messages[1]["content"]
    assert "不要只替换近义词" in revision_system
    assert "用户确认的真实补充事实" in revision_user
    assert "招聘产品经理" in revision_user
    assert "3个城市" in revision_user
    assert "不要添加【请补充真实信息】" in revision_system

    match_messages = build_match_messages("负责产品规划", "招聘产品经理，要求数据分析")
    match_prompt = match_messages[0]["content"]
    assert "核心职责、硬性条件、优先条件" in match_prompt
    assert "未提及不等于候选人没有该能力" in match_prompt
    assert "可直接替换的岗位定向改写示例" in match_prompt

