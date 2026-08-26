"""
==============================================================
 web_app.py — السيرفر الموحّد (للنشر أونلاين)  ·  Google Gemini
==============================================================

هذا الملف يجمع كل شي بسيرفر واحد (Flask):
  - يخدم الموقع كامل (الداشبورد) على "/"
  - يخدم محادثة سارة على "/chat"

بخلاف chat_server.py (اللي يحتاج تشغيل يدوي على جهازك + ملف
منفصل للواجهة)، هذا الملف مصمم للنشر أونلاين على منصة استضافة
(مثل PythonAnywhere) بحيث يصير عندكم رابط واحد حقيقي يشتغل من أي
جهاز، بدون ما تحتاجون تشغّلون أي شي يدوياً وقت العرض.

كل الذكاء الاصطناعي بهذا المشروع على Google Gemini فقط.

المتطلبات (requirements.txt):
    flask
    google-genai

متغيّر البيئة المطلوب (يُضبط من لوحة تحكم الاستضافة، مو بالكود):
    GEMINI_API_KEY

التشغيل محلياً للتجربة (اختياري):
    python3 -m pip install flask google-genai
    export GEMINI_API_KEY=مفتاحكم    (أو حطيه بملف config.txt)
    ./.venv/bin/python web_app.py
    افتحي المتصفح على: http://localhost:5050

ملاحظة: نستخدم المنفذ 5050 مو 5000، لأن macOS يحجز 5000 لخدمة
AirPlay Receiver ويرجّع 403. لتغييره: متغيّر بيئة اسمه PORT.
"""

import json
import os
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify, Response

from app_agent import (
    load_students, run_rules_only_on_student, to_app_student, INPUT_CSV,
    generate_weekly_report, get_gemini_client, LLM_MODEL, friendly_gemini_error,
    tool_fetch_student_records, tool_calc_academic_trend,
    tool_calc_behavior_severity, tool_calc_correlation,
)

BASE_DIR = Path(__file__).parent
TEMPLATE_PATH = BASE_DIR / "app_template.html"
MANUAL_NOTES_PATH = BASE_DIR / "manual_notes.json"
REPORT_STATUS_PATH = BASE_DIR / "report_status.json"

app = Flask(__name__)


# ------------------------------------------------------------
# تخزين بسيط بملفات JSON (كافٍ لعرض تجريبي، بدون قاعدة بيانات)
# ------------------------------------------------------------

def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manual_notes():
    return read_json(MANUAL_NOTES_PATH, {})


def load_report_status():
    return read_json(REPORT_STATUS_PATH, {"status": "قيد المراجعة", "notes": ""})


# ------------------------------------------------------------
# عميل Gemini — المفتاح من متغيرات البيئة أولاً (وضع النشر
# أونلاين)، وإذا ما لقاه يجرب config.txt (وضع التجربة المحلية)
# ------------------------------------------------------------

_client = get_gemini_client()

try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None
    _client = None


# ------------------------------------------------------------
# الصفحة الرئيسية — تولّد الداشبورد بنفس منطق app_agent.py
#
# ملاحظة: هنا نستخدم القواعد الحتمية (بدون استدعاء Gemini) لأن
# الصفحة تُبنى من جديد مع كل زيارة، و٣٠ استدعاء LLM لكل زيارة
# بيكون بطيء ومكلف. تحليل Gemini الكامل يشتغل من app_agent.py
# (شغّليه لما تتغير البيانات)، وسارة بالشات تستخدم Gemini مباشرة.
# ------------------------------------------------------------

@app.route("/")
def home():
    students = load_students(INPUT_CSV)
    manual_notes = load_manual_notes()

    results = []
    for case in students.values():
        outcome = run_rules_only_on_student(case)
        app_s = to_app_student(case, outcome)

        # دمج الملاحظات اليدوية اللي أضافها الأخصائي (لم ترصدها الكاميرا)
        extra = manual_notes.get(case.student_id, [])
        for n in extra:
            label = f"✍️ {n['category']}: {n['note']}"
            if label not in app_s["tags"]:
                app_s["tags"].append(label)

        results.append(app_s)

    order = {"red": 0, "orange": 1, "green": 2}
    results.sort(key=lambda s: order[s["statusKey"]])

    payload = {
        "students": results,
        "report": generate_weekly_report(results, students),
        "report_status": load_report_status(),
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__STUDENTS_DATA__", json.dumps(payload, ensure_ascii=False))
    return Response(html, mimetype="text/html")


# ------------------------------------------------------------
# إضافة ملاحظة يدوية لطالب (حالة لم ترصدها الكاميرا)
# ------------------------------------------------------------

@app.route("/add-note", methods=["POST"])
def add_note():
    body = request.get_json(force=True) or {}
    student_id = body.get("student_id", "")
    note = (body.get("note") or "").strip()
    category = body.get("category", "ملاحظة عامة")

    if not student_id or not note:
        return jsonify({"ok": False, "error": "الرجاء إدخال نص الملاحظة."}), 400

    notes = load_manual_notes()
    notes.setdefault(student_id, []).append({"note": note, "category": category})
    write_json(MANUAL_NOTES_PATH, notes)

    return jsonify({"ok": True})


# ------------------------------------------------------------
# إجراءات الأخصائي على تقرير الوكيل: قبول / رفض / تعديل
# ------------------------------------------------------------

@app.route("/report-action", methods=["POST"])
def report_action():
    body = request.get_json(force=True) or {}
    action = body.get("action")

    status = load_report_status()
    if action == "accept":
        status["status"] = "مقبول"
    elif action == "reject":
        status["status"] = "مرفوض"
    elif action == "edit":
        status["notes"] = (body.get("notes") or "").strip()
        status["status"] = "معدَّل"
    else:
        return jsonify({"ok": False, "error": "إجراء غير معروف"}), 400

    write_json(REPORT_STATUS_PATH, status)
    return jsonify({"ok": True, "status": status})


# ------------------------------------------------------------
# محادثة سارة — نفس منطق chat_server.py بالضبط (Gemini)
# ------------------------------------------------------------

SYSTEM_PROMPT = """You are "Sarah" (سارة), an AI agent inside "البصيرة الرقمية" —
a system that helps school specialists spot early behavioral and academic
changes in students. You are not a simple chatbot: you have real tools to
look up live student data, and you decide yourself, step by step, whether
and when to use them before answering.

Language rule: always reply in the SAME language the specialist just wrote
in (Arabic or English). Do not mix languages in one reply.

Rules:
- Never give a medical or psychological diagnosis for any student.
- Always remind the specialist that the final intervention decision is
  theirs, not yours — you only support and inform.
- Be concise and professional.
- If asked about a student not found via list_students, say clearly that
  there is no data for them.
- Use fetch_student_records, calc_academic_trend, calc_behavior_severity,
  and calc_correlation yourself when a question needs real numbers —
  don't guess or make up data."""


def build_tools(students):
    def list_students() -> dict:
        """Returns the list of all available students (id, name, class)."""
        return {
            "students": [
                {"id": c.student_id, "name": c.name, "class_name": c.class_name}
                for c in students.values()
            ]
        }

    def fetch_student_records(student_id: str) -> dict:
        """Fetches the full weekly record (scores + behavioral notes) for one student, given their id from list_students."""
        return tool_fetch_student_records(student_id, students)

    def calc_academic_trend(scores: List[float]) -> dict:
        """Calculates the academic score trend across weeks (declining/stable/improving) from a list of numeric scores."""
        return tool_calc_academic_trend(scores)

    def calc_behavior_severity(categories: List[str]) -> dict:
        """Calculates a behavioral severity score from a list of category strings (نفسي/صحي/سلوكي)."""
        return tool_calc_behavior_severity(categories)

    def calc_correlation(trend: str, behavior_score: int) -> dict:
        """Determines whether an academic decline is actually correlated with a behavioral signal."""
        return tool_calc_correlation(trend, behavior_score)

    return [list_students, fetch_student_records, calc_academic_trend,
            calc_behavior_severity, calc_correlation]


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    user_message = body.get("message", "")
    history = body.get("history", [])

    if _client is None:
        reply = ("تعذر الاتصال بالنموذج اللغوي. تأكدي أن GEMINI_API_KEY مضبوط "
                  "بإعدادات الاستضافة (Environment Variables)، ثم أعيدي تحميل التطبيق.")
        return jsonify({"reply": reply})

    students = load_students(INPUT_CSV)
    tools = build_tools(students)

    contents = []
    for h in history:
        role = "model" if h.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": h.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=tools,
            ),
        )
        reply = (response.text or "...").strip()
    except Exception as exc:
        reply = friendly_gemini_error(exc)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    # المنفذ 5000 محجوز على macOS لخدمة AirPlay Receiver (يرجّع 403)،
    # عشان كذا الافتراضي عندنا 5050.
    port = int(os.environ.get("PORT", 5050))

    if _client is None:
        print("! تحذير: GEMINI_API_KEY غير موجود — سارة لن ترد فعلياً محلياً.")
    else:
        print(f"✓ Gemini جاهز — النموذج: {LLM_MODEL}")
    print(f"السيرفر شغّال محلياً على: http://localhost:{port}")
    app.run(debug=True, port=port)
