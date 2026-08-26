"""
==============================================================
 خادم "سارة" — AI Agent حقيقي عبر Google Gemini (مجاني، بدون بطاقة بنكية)
==============================================================

وكيل حقيقي (أدوات حقيقية + قرار ذاتي أثناء المحادثة) يشتغل على
Google Gemini عبر Google AI Studio — مفتاح API مجاني بحصة يومية
سخية بدون أي حاجة لبطاقة بنكية.

كل الذكاء الاصطناعي بهذا المشروع على Gemini فقط، بنفس المفتاح
وبنفس النموذج المعرّف بملف app_agent.py.

طريقة الحصول على المفتاح:
    1) روحي aistudio.google.com
    2) سجّلي دخول بحساب Google عادي (مجاني بالكامل)
    3) من "Get API key" ← "Create API key"
    4) انسخي المفتاح (يبدأ عادة بـ AIza...)
    5) حطيه بملف config.txt بنفس المجلد:
       GEMINI_API_KEY=المفتاح_هنا

طريقة التشغيل:
    python3 -m pip install google-genai
    python3 chat_server.py

ملاحظة صراحة: هذا الكود مكتوب حسب توثيق Google الرسمي الحالي،
لكن ما قدرنا نختبره حياً بمفتاح فعلي. لو طلع أي خطأ عند التشغيل،
ابعتيه لنا بالنص كامل عشان نصلحه بسرعة.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

from app_agent import (
    load_students, INPUT_CSV, get_gemini_client, LLM_MODEL, friendly_gemini_error,
    tool_fetch_student_records, tool_calc_academic_trend,
    tool_calc_behavior_severity, tool_calc_correlation,
)

PORT = 8787


# ------------------------------------------------------------
# عميل Gemini — المفتاح يُقرأ من متغيرات البيئة أو من config.txt
# (نفس الدالة المستخدمة في app_agent.py و web_app.py)
# ------------------------------------------------------------

_client = get_gemini_client()

try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None
    _client = None


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
    """نبني الأدوات هنا (مو بمستوى الملف) عشان كل أداة تقدر توصل
    لبيانات الطلاب الحالية عبر closure، بدون تمرير مفتاح إضافي
    يدوياً — الـ SDK يستدعي هذي الدوال تلقائياً بنفسه."""

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


def run_chat_turn(user_message: str, history: list) -> str:
    if _client is None:
        return ("تعذر الاتصال بالنموذج اللغوي. تأكدي أن ملف config.txt فيه "
                "GEMINI_API_KEY صحيح (يبدأ بـ AIza) بنفس مجلد chat_server.py، "
                "ثم أعيدي تشغيل السيرفر.")

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
        return (response.text or "...").strip()
    except Exception as exc:
        return friendly_gemini_error(exc)


class ChatHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        user_message = body.get("message", "")
        history = body.get("history", [])

        try:
            reply = run_chat_turn(user_message, history)
        except Exception as exc:
            reply = friendly_gemini_error(exc)

        payload = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print("[chat_server]", fmt % args)


def run():
    if _client is not None:
        print(f"✓ Gemini جاهز — النموذج: {LLM_MODEL}")
    if _client is None:
        print("! تحذير: GEMINI_API_KEY غير موجود أو فارغ.")
        print("  افتحي ملف config.txt بنفس المجلد وحطي فيه سطر (بدون علامات تنصيص):")
        print("  GEMINI_API_KEY=مفتاحك_الحقيقي")
        print("  احصلي عليه مجاناً من: https://aistudio.google.com")
        print("  ثم شغّلي السيرفر من جديد.\n")

    server = HTTPServer(("localhost", PORT), ChatHandler)
    print(f"سارة (AI Agent · Gemini) جاهزة على http://localhost:{PORT}  —  اتركي هذه النافذة مفتوحة.")
    server.serve_forever()


if __name__ == "__main__":
    run()
