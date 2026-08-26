"""
==============================================================
 AI Agent (Agentic Loop) — محرّك app.html  ·  Google Gemini
==============================================================

هذا الملف هو "العقل" الحقيقي وراء app.html. بدل ما تكون بيانات
الطلاب مكتوبة يدوياً بجافاسكربت، هذا السكربت:

  1) يقرأ بيانات الطلاب الأسبوعية من CSV (سلوك + درجات)
  2) يشغّل وكيل ذكاء اصطناعي حقيقي (Google Gemini، عبر Function
     Calling) يحلل كل طالب بنفسه: يقرر بنفسه أي أداة يستخدم
     ومتى، ثم يقدّم استنتاجه النهائي
  3) شبكة أمان حتمية (rules) تتحقق من قرار الوكيل ولا تسمح
     بتخفيض التصعيد في الحالات الحرجة
  4) يحقن النتيجة داخل app_template.html في مكان __STUDENTS_DATA__
     وينتج app.html جاهز للعرض

النظام كامل يعمل على Gemini فقط — لا يوجد أي اعتماد على مزوّد آخر.

المفتاح: GEMINI_API_KEY
  - إما من متغيرات البيئة (وضع النشر أونلاين)
  - أو من ملف config.txt بنفس المجلد (وضع التجربة المحلية)

بدون مفتاح API: يعمل السكربت بالقواعد الحتمية فقط (بدون ملاحظات
LLM نصية)، فتبقى الواجهة شغالة بالكامل، لكن بدون "البصيرة الذكية"
المولّدة فعلياً من النموذج اللغوي.
"""

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "students_data_elementary.csv"
TEMPLATE_HTML = BASE_DIR / "app_template.html"
OUTPUT_HTML = BASE_DIR / "app.html"

# النموذج المستخدم في كل أجزاء النظام (المحرك + شات سارة).
# اخترنا flash-lite لأنه سريع (٤ ثواني للطالب الواحد) وكافٍ تماماً
# لمهام الأدوات هذي. بدائل مجرَّبة بنفس المفتاح:
#   "gemini-3.5-flash"  → أذكى بس أبطأ بكثير (٤٧ ثانية للطالب)
#   "gemini-3.7-flash"  → الأحدث، أحياناً مزحوم ويرجّع 503
# للتغيير: عدّلي السطر تحت، أو حطي متغيّر بيئة GEMINI_MODEL
LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_AGENT_STEPS = 8

# الحصة المجانية من Google محدودة (١٥ طلب بالدقيقة للنموذج الواحد).
# كل طالب يستهلك ٤-٥ طلبات، فنحتاج ننتظر ونعيد المحاولة بدل ما
# نطيح على القواعد الحتمية ونخسر ملاحظات الوكيل.
MAX_RETRIES = 6
PACE_SECONDS = 4.0   # فاصل بسيط بين الطلاب عشان ما نضرب السقف من أول

CATEGORY_WEIGHT = {"نفسي": 3, "صحي": 3, "سلوكي": 2}
DECLINE_STRONG = -3
DECLINE_MILD = -1
ESCALATE_BEHAVIOR_SCORE = 6
MEDIUM_BEHAVIOR_SCORE = 3

LEVEL_TO_STATUS = {"عالي": "red", "متوسط": "orange", "منخفض": "green"}
STATUS_LABEL = {"red": "🔴 يحتاج تدخل", "orange": "🟠 يحتاج متابعة", "green": "🟢 طبيعي"}
LEVEL_RANK = {"منخفض": 0, "متوسط": 1, "عالي": 2}


# ------------------------------------------------------------
# مفتاح Gemini — مصدر واحد يستخدمه كل ملفات المشروع
# ------------------------------------------------------------

def load_local_config() -> None:
    """يقرأ config.txt (أو .env / env.txt) ويحط القيم بمتغيرات البيئة.
    متغيرات البيئة الموجودة أصلاً لها الأولوية (وضع النشر أونلاين)."""
    for filename in ["config.txt", ".env", "env.txt"]:
        path = BASE_DIR / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def api_key_looks_valid(api_key: str) -> bool:
    """يرفض المفتاح الافتراضي اللي بملف config.txt (نص عربي توضيحي)."""
    return len(api_key) > 15 and "ضعي" not in api_key and "مفتاح" not in api_key


def get_gemini_client():
    """يرجّع عميل Gemini جاهز، أو None إذا ما فيه مفتاح صالح أو
    مكتبة google-genai غير مثبّتة."""
    load_local_config()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key_looks_valid(api_key):
        return None
    try:
        from google import genai
    except ImportError:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


# ------------------------------------------------------------
# نماذج البيانات
# ------------------------------------------------------------

@dataclass
class WeeklyRecord:
    week: int
    academic_score: float
    behavior_flag: bool
    behavior_category: str
    behavior_note: str


@dataclass
class StudentCase:
    student_id: str
    name: str
    class_name: str
    records: List[WeeklyRecord] = field(default_factory=list)


def load_students(csv_path: Path) -> Dict[str, StudentCase]:
    students: Dict[str, StudentCase] = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row["student_id"]
            if sid not in students:
                students[sid] = StudentCase(sid, row["name"], row["class_name"])
            students[sid].records.append(WeeklyRecord(
                week=int(row["week"]),
                academic_score=float(row["academic_score"]),
                behavior_flag=row["behavior_flag"] == "1",
                behavior_category=row["behavior_category"],
                behavior_note=row["behavior_note"],
            ))
    for case in students.values():
        case.records.sort(key=lambda r: r.week)
    return students


# ------------------------------------------------------------
# الأدوات (Tools) التي يستخدمها الوكيل بقراره الخاص
# ------------------------------------------------------------

def tool_fetch_student_records(student_id: str, students: Dict[str, StudentCase]) -> dict:
    case = students.get(student_id)
    if not case:
        return {"error": f"لا يوجد طالب بالمعرف {student_id}"}
    return {
        "name": case.name,
        "class_name": case.class_name,
        "weeks": [r.week for r in case.records],
        "scores": [r.academic_score for r in case.records],
        "behavior_events": [
            {"week": r.week, "category": r.behavior_category, "note": r.behavior_note}
            for r in case.records if r.behavior_flag
        ],
    }


def tool_calc_academic_trend(scores: List[float]) -> dict:
    n = len(scores)
    if n < 2:
        return {"slope": 0.0, "trend": "غير كافٍ للتحليل"}
    weeks = list(range(1, n + 1))
    mean_w, mean_s = mean(weeks), mean(scores)
    num = sum((w - mean_w) * (s - mean_s) for w, s in zip(weeks, scores))
    den = sum((w - mean_w) ** 2 for w in weeks)
    slope = num / den if den else 0.0

    if slope <= DECLINE_STRONG:
        trend = "تراجع واضح"
    elif slope <= DECLINE_MILD:
        trend = "تراجع طفيف"
    elif slope < 1:
        trend = "مستقر"
    else:
        trend = "تحسّن"
    return {"slope": round(slope, 2), "trend": trend}


def tool_calc_behavior_severity(categories: List[str]) -> dict:
    return {"behavior_score": sum(CATEGORY_WEIGHT.get(c, 1) for c in categories)}


def tool_calc_correlation(trend: str, behavior_score: int) -> dict:
    declining = trend in ("تراجع واضح", "تراجع طفيف")
    if declining and behavior_score > 0:
        status = "مرتبطة"
    elif declining and behavior_score == 0:
        status = "تراجع أكاديمي بدون مؤشر سلوكي واضح"
    elif not declining and behavior_score > 0:
        status = "مؤشر سلوكي دون تأثير أكاديمي ظاهر"
    else:
        status = "لا توجد مؤشرات"
    return {"correlation": status}


# صيغة Gemini Function Declarations (أنواع OpenAPI بحروف كبيرة)
TOOL_DECLARATIONS = [
    {
        "name": "fetch_student_records",
        "description": "يجلب السجل الأسبوعي الكامل لطالب معيّن: الدرجات والملاحظات السلوكية.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"student_id": {"type": "STRING"}},
            "required": ["student_id"],
        },
    },
    {
        "name": "calc_academic_trend",
        "description": "يحسب اتجاه الدرجات الأكاديمية عبر الأسابيع ويصنّفه.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"scores": {"type": "ARRAY", "items": {"type": "NUMBER"}}},
            "required": ["scores"],
        },
    },
    {
        "name": "calc_behavior_severity",
        "description": "يحسب درجة الخطورة السلوكية بناءً على فئات الملاحظات (نفسي/صحي/سلوكي).",
        "parameters": {
            "type": "OBJECT",
            "properties": {"categories": {"type": "ARRAY", "items": {"type": "STRING"}}},
            "required": ["categories"],
        },
    },
    {
        "name": "calc_correlation",
        "description": "يحدد هل التراجع الأكاديمي مرتبط فعلياً بمؤشر سلوكي أم لا.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"trend": {"type": "STRING"}, "behavior_score": {"type": "INTEGER"}},
            "required": ["trend", "behavior_score"],
        },
    },
    {
        "name": "submit_case_conclusion",
        "description": "قدّم الاستنتاج النهائي لحالة الطالب بعد اكتمال التحليل. الخطوة الأخيرة الوحيدة.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "alert_level": {"type": "STRING", "enum": ["عالي", "متوسط", "منخفض"]},
                "correlation": {"type": "STRING"},
                "recommendation": {"type": "STRING"},
                "insight": {"type": "STRING", "description": "ملاحظة مهنية موجزة (سطرين) للأخصائي"},
            },
            "required": ["alert_level", "correlation", "recommendation", "insight"],
        },
    },
]

TOOL_IMPLS = {
    "calc_academic_trend": lambda args, students: tool_calc_academic_trend(args["scores"]),
    "calc_behavior_severity": lambda args, students: tool_calc_behavior_severity(args["categories"]),
    "calc_correlation": lambda args, students: tool_calc_correlation(args["trend"], args["behavior_score"]),
    "fetch_student_records": lambda args, students: tool_fetch_student_records(args["student_id"], students),
}

SYSTEM_PROMPT = """أنتِ وكيل ذكاء اصطناعي يساعد أخصائية اجتماعية/نفسية في مدرسة ابتدائية
على رصد حالات الطلبة. مهمتك تحليل حالة طالب واحد بكل استقلالية باستخدام الأدوات المتاحة.

خطوات مقترحة (لكِ حرية الترتيب): اجلبي بيانات الطالب، احسبي اتجاه درجاته، احسبي
شدة المؤشرات السلوكية، احسبي الترابط بينهما، ثم قدّمي استنتاجك النهائي عبر
submit_case_conclusion.

قواعد إلزامية عند تحديد alert_level:
- "عالي": إذا كان الترابط = "مرتبطة" والاتجاه = "تراجع واضح" وشدة السلوك >= 6.
- "متوسط": إذا وُجد ترابط، أو تراجع (واضح أو طفيف)، أو شدة سلوك >= 3.
- "منخفض": غير ذلك.

اكتبي insight بأسلوب موجز مناسب لأعمار الأطفال (٦-١٢ سنة)، ولا تُصدري تشخيصاً طبياً
أو نفسياً. القرار النهائي بالتدخل يبقى للأخصائية البشرية دائماً."""


RETRYABLE = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "429", "503")


def is_retryable(exc) -> bool:
    return any(k in str(exc) for k in RETRYABLE)


def friendly_gemini_error(exc) -> str:
    """رسالة عربية مفهومة بدل ما نطبع JSON الخطأ كامل للأخصائية."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return ("الحصة المجانية لهذه الدقيقة امتلأت (١٥ طلب بالدقيقة). "
                "انتظري دقيقة وجربي مرة ثانية.")
    if "UNAVAILABLE" in text or "503" in text:
        return "النموذج مزحوم حالياً عند Google. جربي بعد لحظات."
    if "API_KEY_INVALID" in text or "PERMISSION_DENIED" in text or "401" in text:
        return "مفتاح GEMINI_API_KEY غير صالح أو منتهي. تأكدي منه بملف config.txt."
    return f"صار خطأ أثناء الاتصال بـ Gemini: {text[:200]}"


def _retry_delay(exc, attempt: int) -> float:
    """نحترم المدة اللي يطلبها Google نفسه، وإذا ما ذكرها نضاعف تدريجياً."""
    match = re.search(r"'retryDelay': '(\d+)s'", str(exc))
    if match:
        return int(match.group(1)) + 2
    return min(60.0, 5.0 * (2 ** attempt))


def generate_with_retry(client, contents, config):
    """استدعاء Gemini مع صبر على حدود الحصة المجانية."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=LLM_MODEL, contents=contents, config=config,
            )
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            delay = _retry_delay(exc, attempt)
            print(f"    ⏳ حد الحصة المجانية — انتظار {delay:.0f} ثانية ثم إعادة المحاولة "
                  f"({attempt + 1}/{MAX_RETRIES})...", flush=True)
            time.sleep(delay)
    raise last_exc


def deterministic_floor(trend: str, behavior_score: int, correlation: str) -> str:
    if correlation == "مرتبطة" and trend == "تراجع واضح" and behavior_score >= ESCALATE_BEHAVIOR_SCORE:
        return "عالي"
    if correlation == "مرتبطة" or trend in ("تراجع واضح", "تراجع طفيف") or behavior_score >= MEDIUM_BEHAVIOR_SCORE:
        return "متوسط"
    return "منخفض"


def compute_ground_truth(case: StudentCase) -> Dict:
    """يحسب الاتجاه/الشدة/الترابط مباشرة من البيانات، بدون أي LLM.
    تستخدمه شبكة الأمان لما الوكيل يتخطّى الأدوات ويقفز للاستنتاج."""
    scores = [r.academic_score for r in case.records]
    trend = tool_calc_academic_trend(scores)["trend"]
    categories = [r.behavior_category for r in case.records if r.behavior_flag]
    behavior_score = tool_calc_behavior_severity(categories)["behavior_score"]
    correlation = tool_calc_correlation(trend, behavior_score)["correlation"]
    return {"trend": trend, "behavior_score": behavior_score, "correlation": correlation}


def run_agent_on_student(client, student_id: str, students: Dict[str, StudentCase]) -> Dict:
    """حلقة الوكيل الحقيقية على Gemini: النموذج يختار الأدوات بنفسه،
    ونحن ننفّذها ونرجّع له النتيجة، لين يقدّم استنتاجه النهائي."""
    from google.genai import types

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"حلّلي حالة الطالب صاحب المعرف {student_id}.")],
    )]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        # نعطّل التنفيذ التلقائي عشان نمسك كل استدعاء أداة بأنفسنا
        # (نسجّله بالـ log ونطبّق شبكة الأمان عليه)
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0,
    )

    tool_calls_log: List[str] = []
    # القيم الحقيقية من البيانات — شبكة الأمان تعتمد عليها حتى لو
    # الوكيل ما استدعى ولا أداة
    truth = compute_ground_truth(students[student_id])
    last_trend = truth["trend"]
    last_behavior_score = truth["behavior_score"]
    last_correlation = truth["correlation"]

    for _ in range(MAX_AGENT_STEPS):
        response = generate_with_retry(client, contents, config)

        calls = response.function_calls or []
        if not calls:
            break

        contents.append(response.candidates[0].content)

        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            tool_calls_log.append(call.name)

            if call.name == "submit_case_conclusion":
                floor = deterministic_floor(last_trend, last_behavior_score, last_correlation)
                final_level = args.get("alert_level", floor)
                if final_level not in LEVEL_RANK:
                    final_level = floor
                overridden = LEVEL_RANK[floor] > LEVEL_RANK[final_level]
                if overridden:
                    final_level = floor
                return {
                    "alert_level": final_level,
                    "correlation": args.get("correlation", last_correlation),
                    "recommendation": args.get("recommendation", ""),
                    "llm_insight": args.get("insight"),
                    "trend": last_trend, "behavior_score": last_behavior_score,
                    "safety_overridden": overridden, "tool_calls": tool_calls_log,
                }

            impl = TOOL_IMPLS.get(call.name)
            if impl is None:
                result = {"error": f"أداة غير معروفة: {call.name}"}
            else:
                try:
                    result = impl(args, students)
                except Exception as exc:
                    result = {"error": f"فشل تنفيذ الأداة: {exc}"}

            if call.name == "calc_academic_trend" and "trend" in result:
                last_trend = result["trend"]
            elif call.name == "calc_behavior_severity" and "behavior_score" in result:
                last_behavior_score = result["behavior_score"]
            elif call.name == "calc_correlation" and "correlation" in result:
                last_correlation = result["correlation"]

            result_parts.append(types.Part.from_function_response(
                name=call.name, response=result,
            ))

        contents.append(types.Content(role="user", parts=result_parts))

    floor = deterministic_floor(last_trend, last_behavior_score, last_correlation)
    return {
        "alert_level": floor, "correlation": last_correlation,
        "recommendation": "لم يُكمل الوكيل تحليله ضمن الحد المسموح؛ يُعرض الحد الأدنى الآمن ويلزم مراجعة يدوية.",
        "llm_insight": None, "trend": last_trend, "behavior_score": last_behavior_score,
        "safety_overridden": True, "tool_calls": tool_calls_log,
    }


def run_rules_only_on_student(case: StudentCase) -> Dict:
    """مسار احتياطي بدون LLM: نفس المنطق لكن محسوب مباشرة بالكود، بدون استدعاء API."""
    truth = compute_ground_truth(case)
    level = deterministic_floor(truth["trend"], truth["behavior_score"], truth["correlation"])

    recs = {
        "عالي": "تصعيد فوري للأخصائية للتدخل المباشر مع الأسرة.",
        "متوسط": "تنبيه ومتابعة أسبوعية دون تصعيد فوري.",
        "منخفض": "لا حاجة للتدخل حالياً؛ يكتفى بالأرشفة.",
    }
    return {
        "alert_level": level, "correlation": truth["correlation"], "recommendation": recs[level],
        "llm_insight": None, "trend": truth["trend"], "behavior_score": truth["behavior_score"],
        "safety_overridden": False, "tool_calls": ["rules_only"],
    }


# ------------------------------------------------------------
# تحويل النتيجة لصيغة app.html
# ------------------------------------------------------------

TREND_TO_PERFORMANCE_LABEL = {
    "تراجع واضح": "📉 متراجع", "تراجع طفيف": "📉 متراجع",
    "مستقر": "→ مستقر", "تحسّن": "↑ تحسن", "غير كافٍ للتحليل": "→ مستقر",
}


def to_app_student(case: StudentCase, outcome: Dict) -> Dict:
    status = LEVEL_TO_STATUS[outcome["alert_level"]]
    last_score = case.records[-1].academic_score if case.records else None
    notes = sorted({r.behavior_note for r in case.records if r.behavior_flag and r.behavior_note})

    insight = outcome["llm_insight"] or outcome["recommendation"]

    weekly = [
        {
            "week": r.week, "score": r.academic_score, "flagged": r.behavior_flag,
            "category": r.behavior_category, "note": r.behavior_note,
        }
        for r in case.records
    ]

    return {
        "id": case.student_id,
        "name": case.name,
        "className": case.class_name,
        "score": f"{int(last_score)}%" if last_score is not None else "—",
        "statusKey": status,
        "statusLabel": STATUS_LABEL[status],
        "performanceLabel": TREND_TO_PERFORMANCE_LABEL.get(outcome["trend"], "→ مستقر"),
        "tags": notes if notes else ["✅ لا توجد مؤشرات سلوكية"],
        "insight": insight,
        "weekly": weekly,
    }


def generate_weekly_report(results: List[Dict], students: Dict[str, "StudentCase"]) -> Dict:
    """تقرير الوكيل الحالي — يستخدمه app_agent.py و web_app.py معاً
    عشان النسخة المولّدة محلياً والنسخة المنشورة أونلاين يطلعون بنفس
    شكل البيانات اللي يتوقعه القالب.

    يحتوي أيضاً على السجل الأسبوعي الكامل (الدرجات + كل ملاحظة
    سلوكية) لكل طالب مذكور بالتقرير (عالي/متوسط)، عشان الأخصائية
    تشوف تسلسل الحالة أسبوعاً بأسبوع، مو ملخصاً واحداً بس."""
    from datetime import date

    def weekly_rows(student_id: str):
        case = students.get(student_id)
        if not case:
            return []
        return [
            {
                "week": r.week,
                "score": r.academic_score,
                "flagged": r.behavior_flag,
                "category": r.behavior_category,
                "note": r.behavior_note,
            }
            for r in case.records
        ]

    high = [r for r in results if r["statusKey"] == "red"]
    medium = [r for r in results if r["statusKey"] == "orange"]
    low = [r for r in results if r["statusKey"] == "green"]

    if high:
        summary = f"{len(high)} حالة تحتاج تدخلاً فورياً، و{len(medium)} تحتاج متابعة."
    elif medium:
        summary = f"لا توجد حالات حرجة، لكن {len(medium)} حالة تحتاج متابعة."
    else:
        summary = "جميع الطلاب ضمن المستوى الطبيعي هذا الأسبوع."

    return {
        "date": date.today().isoformat(),
        "title": "تقرير الوكيل الأسبوعي",
        "summary": summary,
        "high": [
            {"name": r["name"], "class_name": r["className"], "insight": r["insight"], "weekly": weekly_rows(r["id"])}
            for r in high
        ],
        "medium": [
            {"name": r["name"], "class_name": r["className"], "insight": r["insight"], "weekly": weekly_rows(r["id"])}
            for r in medium
        ],
        "low_count": len(low),
        "total": len(results),
    }


def load_manual_notes_local() -> Dict:
    path = BASE_DIR / "manual_notes.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_report_status_local() -> Dict:
    path = BASE_DIR / "report_status.json"
    if not path.exists():
        return {"status": "قيد المراجعة", "notes": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "قيد المراجعة", "notes": ""}


# ------------------------------------------------------------
# التشغيل الكامل
# ------------------------------------------------------------

def run() -> None:
    students = load_students(INPUT_CSV)

    client = get_gemini_client()

    if client is None:
        print("! لم يتم العثور على GEMINI_API_KEY صالح أو مكتبة google-genai.")
        print("  سيعمل الوكيل بالقواعد الحتمية فقط (بدون ملاحظات LLM نصية).")
        print("  للتفعيل الكامل: python3 -m pip install google-genai")
        print("                  ثم حطي مفتاحكم بملف config.txt:")
        print("                  GEMINI_API_KEY=المفتاح_هنا")
        print("  المفتاح مجاني من: https://aistudio.google.com\n")
    else:
        print(f"✓ Gemini جاهز — النموذج: {LLM_MODEL}\n")

    print("=" * 55)
    print(" تشغيل AI Agent لكل طالب (ابتدائي)")
    print("=" * 55)

    app_students = []
    manual_notes = load_manual_notes_local()

    for case in students.values():
        if client is not None:
            try:
                outcome = run_agent_on_student(client, case.student_id, students)
            except Exception as exc:
                print(f"  ! تعذّر تحليل {case.name} عبر Gemini: {friendly_gemini_error(exc)}")
                print("    تم استخدام القواعد الحتمية لهذه الحالة.")
                outcome = run_rules_only_on_student(case)
            time.sleep(PACE_SECONDS)
        else:
            outcome = run_rules_only_on_student(case)

        override_note = "  (تصعيد تلقائي عبر شبكة الأمان)" if outcome["safety_overridden"] else ""
        print(f"- {case.name} ({case.class_name}): {outcome['alert_level']}{override_note} "
              f"| خطوات: {' → '.join(outcome['tool_calls'])}")

        app_s = to_app_student(case, outcome)
        extra = manual_notes.get(case.student_id, [])
        for n in extra:
            label = f"✍️ {n['category']}: {n['note']}"
            if label not in app_s["tags"]:
                app_s["tags"].append(label)

        app_students.append(app_s)

    order = {"red": 0, "orange": 1, "green": 2}
    app_students.sort(key=lambda s: order[s["statusKey"]])

    payload = {
        "students": app_students,
        "report": generate_weekly_report(app_students, students),
        "report_status": load_report_status_local(),
    }

    template = TEMPLATE_HTML.read_text(encoding="utf-8")
    html = template.replace("__STUDENTS_DATA__", json.dumps(payload, ensure_ascii=False))
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print(f"\nتم إنشاء التطبيق: {OUTPUT_HTML}")


if __name__ == "__main__":
    run()
