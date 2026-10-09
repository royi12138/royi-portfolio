from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

from app.ai_analysis import analyze_resume_text

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TMP_DIR = BASE_DIR.parent / "tmp"
TMP_DIR.mkdir(exist_ok=True)

MAX_BYTES = 10 * 1024 * 1024
MAX_ANALYSIS_CHARS = 20_000
SUPPORTED = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}

app = FastAPI(title="AI 岗位匹配简历助手", version="0.2.0")

SECTION_ALIASES = {
    "basic_info": {"个人信息", "基本信息", "个人资料", "联系方式", "contact", "profile"},
    "education": {"教育经历", "教育背景", "学历", "教育", "education"},
    "work_experience": {"工作经历", "工作经验", "职业经历", "任职经历", "experience", "work experience", "employment"},
    "project_experience": {"项目经历", "项目经验", "项目", "projects", "project experience"},
    "skills": {"技能", "专业技能", "技能特长", "skills", "technical skills"},
    "self_intro": {"自我评价", "个人简介", "summary", "profile summary"},
}


class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(min_length=30, max_length=MAX_ANALYSIS_CHARS)


def safe_filename(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^\w.\-\u4e00-\u9fff ]+", "_", name)
    return name[:120] or "upload"


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in text.split("\n")]
    return "\n".join(x for x in lines if x)


def extract_pdf(path: Path) -> tuple[str, int]:
    doc = fitz.open(path)
    try:
        pages = doc.page_count
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                parts.append(f"[第{i + 1}页]\n{text}")
        return normalize_text("\n\n".join(parts)), pages
    finally:
        doc.close()


def estimate_docx_pages(document: Document, text: str) -> int:
    breaks = 0
    for paragraph in document.paragraphs:
        breaks += paragraph._p.xml.count('w:type="page"')
    estimated = max(1, (len(text) + 1049) // 1050)
    return max(estimated, breaks + 1)


def extract_docx(path: Path) -> tuple[str, int]:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    text = normalize_text("\n".join(parts))
    return text, estimate_docx_pages(doc, text)


def extract_image(path: Path) -> tuple[str, int]:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    try:
        text = pytesseract.image_to_string(image, lang="chi_sim+eng", config="--psm 6")
    except pytesseract.TesseractError:
        text = pytesseract.image_to_string(image, config="--psm 6")
    return normalize_text(text), 1


def section_for_heading(line: str) -> str | None:
    normalized = re.sub(r"[\s:：|·•/\\]+", "", line).lower()
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_norm = re.sub(r"[\s:：|·•/\\]+", "", alias).lower()
            if normalized == alias_norm:
                return section
    if "工作" in line and ("经历" in line or "经验" in line):
        return "work_experience"
    if "项目" in line:
        return "project_experience"
    if "教育" in line or "学历" in line:
        return "education"
    if "技能" in line:
        return "skills"
    return None


def parse_sections(text: str) -> dict[str, list[str]]:
    data = {key: [] for key in SECTION_ALIASES}
    data["other"] = []
    current = "basic_info"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = section_for_heading(line)
        if heading:
            current = heading
        else:
            data[current].append(line)
    return data


def extract_basic_info(text: str, sections: dict[str, list[str]]) -> dict[str, Any]:
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"(?:\+?86[-\s]?)?1\d{10}", text)
    urls = re.findall(r"https?://\S+", text)
    return {
        "possible_email": email[:2],
        "possible_phone": phone[:2],
        "possible_urls": urls[:3],
        "raw_lines": sections.get("basic_info", [])[:12],
    }


def build_resume_json(filename: str, ext: str, text: str, pages: int) -> dict[str, Any]:
    sections = parse_sections(text)
    return {
        "document": {"filename": filename, "type": ext.lstrip("."), "pages": pages},
        "basic_info": extract_basic_info(text, sections),
        "education": sections["education"],
        "work_experience": sections["work_experience"],
        "project_experience": sections["project_experience"],
        "skills": sections["skills"],
        "self_intro": sections["self_intro"],
        "other": sections["other"],
        "raw_text": text,
        "parser": {"mode": "heuristic_mvp", "note": "已接入千问简历分析；语义抽取仍在后续完善。"},
    }


async def save_upload(file: UploadFile) -> Path:
    original = safe_filename(file.filename or "upload")
    suffix = Path(original).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, "暂不支持该文件类型，请上传 PDF、DOCX、JPG 或 PNG。")

    tmp = Path(tempfile.mkdtemp(prefix="resume_", dir=TMP_DIR))
    target = tmp / original
    size = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "文件不能超过 10MB。")
                out.write(chunk)
        return target
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def parse_file(path: Path) -> tuple[str, int]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path)
    if suffix in {".jpg", ".jpeg", ".png"}:
        return extract_image(path)
    raise ValueError("unsupported file")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/api/resume/parse")
async def parse_resume(file: UploadFile = File(...)) -> dict[str, Any]:
    path = await save_upload(file)
    try:
        text, pages = parse_file(path)
        if pages > 3:
            raise HTTPException(400, "目前仅支持 3 页以内的简历，请压缩后重新上传。")
        if len(text.strip()) < 30:
            raise HTTPException(422, "没有识别到足够的简历文字，请上传更清晰的文件。")
        resume = build_resume_json(path.name, path.suffix, text, pages)
        return {"ok": True, "resume": resume}
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.post("/api/resume/analyze")
async def analyze_resume(request: ResumeAnalysisRequest) -> dict[str, Any]:
    analysis = await analyze_resume_text(request.resume_text)
    return {"ok": True, "analysis": analysis}
