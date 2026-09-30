import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

ANALYSIS_FIELDS = [
    "market", "problem", "value", "pricing",
    "strength", "weakness", "opportunity", "threat",
    "marketing", "risks", "score", "scoretext",
    "recommendation", "assumptions",
    "validation_questions", "action_plan"
]

SYSTEM = """You are Vyapaar Sathi, an AI business-validation assistant for BBA students and first-time entrepreneurs.
Analyze a business idea using the supplied customer, location and budget.
Be practical, specific, conservative and concise. Do not guarantee success.
Treat scores and estimates as preliminary.
Return ONLY valid JSON with exactly these fields:
market, problem, value, pricing, strength, weakness, opportunity, threat, marketing, risks, score, scoretext, recommendation, assumptions, validation_questions, action_plan.
score must be an integer from 0 to 100.
assumptions, validation_questions and action_plan must each be arrays of 3 concise strings."""

DEEP_SYNTHESIS_SYSTEM = """You are the final synthesis engine for Vyapaar Sathi.
You receive two independent preliminary business analyses: one from Gemini and one from OpenAI.
Compare them without pretending either is verified market data.
Return ONLY valid JSON with exactly:
consensus, common_strengths, common_concerns, key_disagreement, decision_focus, final_action_plan, final_recommendation.
All list fields must contain 2-4 concise strings.
final_recommendation must be 2-4 sentences."""

def env(name):
    # Render secrets are read directly from the process environment.
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) else ""

def fallback(d):
    idea = d["idea"]
    c = d.get("customer") or "the target customer"
    loc = d.get("location") or "the target location"
    b = d.get("budget") or "the stated budget"
    return {
        "market": f"{c} in {loc}, with demand linked to the need described in '{idea}'.",
        "problem": "The business should solve a specific customer pain point better, cheaper, faster or more conveniently than alternatives.",
        "value": f"A focused offering designed around the core customer need in '{idea}'.",
        "pricing": f"Test multiple price points with real customers. Budget context: {b}.",
        "strength": "Clear positioning and a focused customer segment can simplify early marketing.",
        "weakness": "Demand, operating costs and willingness-to-pay are still unvalidated.",
        "opportunity": "Local partnerships, referrals and a small pilot can create useful early evidence.",
        "threat": "Existing alternatives, price pressure and changing customer preferences.",
        "marketing": "Use local digital channels, referrals, partnerships and a small pilot campaign.",
        "risks": "Weak demand, wrong pricing, operating costs and customer retention.",
        "score": 70,
        "scoretext": "Fallback prototype assessment. Connect Gemini for idea-specific AI analysis.",
        "recommendation": "Interview potential customers, study alternatives, test pricing and run a small pilot before scaling.",
        "assumptions": [
            "Customers have the stated problem",
            "Customers will pay the tested price",
            "The business can deliver at sustainable cost"
        ],
        "validation_questions": [
            "How do you solve this problem today?",
            "What would make you switch?",
            "What price would feel reasonable?"
        ],
        "action_plan": [
            "Interview 10 target customers",
            "Run a small paid pilot",
            "Compare results against cost and retention"
        ]
    }

def clean_json(text):
    if not text:
        raise ValueError("AI returned an empty response")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)

def api_error(resp):
    try:
        data = resp.json()
        err = data.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code") or str(err)
        else:
            msg = str(err)
        return f"HTTP {resp.status_code}: {msg}"[:240]
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:180]}"[:240]

def gemini_analysis(d):
    key = env("GEMINI_API_KEY")
    if not key:
        return None, "GEMINI_API_KEY is missing from the running Render service."

    prompt = f"""Business idea: {d["idea"]}
Target customer: {d.get("customer") or "Not specified"}
Location: {d.get("location") or "Not specified"}
Investment budget: {d.get("budget") or "Not specified"}
Generate the structured preliminary validation report."""

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.25
        }
    }

    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": key},
            json=body,
            timeout=45
        )
        if not r.ok:
            return None, "Gemini request failed: " + api_error(r)

        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = clean_json(raw)
        data["score"] = max(0, min(100, int(data.get("score", 70))))
        return data, None
    except Exception as e:
        return None, f"Gemini error: {str(e)[:180]}"

def openai_analysis(d):
    key = env("OPENAI_API_KEY")
    if not key:
        return None, "OPENAI_API_KEY is missing from the running Render service."

    model = env("OPENAI_MODEL") or "gpt-5.6-luna"

    prompt = f"""Analyze this business idea independently as a skeptical business critic.
Business idea: {d["idea"]}
Target customer: {d.get("customer") or "Not specified"}
Location: {d.get("location") or "Not specified"}
Investment budget: {d.get("budget") or "Not specified"}

Return ONLY JSON with these fields:
market, problem, value, pricing, strength, weakness, opportunity, threat,
marketing, risks, score, scoretext, recommendation, assumptions,
validation_questions, action_plan.
score is illustrative from 0-100.
arrays assumptions, validation_questions and action_plan must each have 3 concise strings."""

    schema = {
        "type": "json_schema",
        "name": "business_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "problem": {"type": "string"},
                "value": {"type": "string"},
                "pricing": {"type": "string"},
                "strength": {"type": "string"},
                "weakness": {"type": "string"},
                "opportunity": {"type": "string"},
                "threat": {"type": "string"},
                "marketing": {"type": "string"},
                "risks": {"type": "string"},
                "score": {"type": "integer"},
                "scoretext": {"type": "string"},
                "recommendation": {"type": "string"},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "validation_questions": {"type": "array", "items": {"type": "string"}},
                "action_plan": {"type": "array", "items": {"type": "string"}}
            },
            "required": ANALYSIS_FIELDS,
            "additionalProperties": False
        }
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                "text": {"format": schema}
            },
            timeout=60
        )

        if not r.ok:
            return None, "OpenAI request failed: " + api_error(r)

        j = r.json()
        raw = j.get("output_text")

        if not raw:
            for item in j.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        raw = part["text"]
                        break
                if raw:
                    break

        data = clean_json(raw)
        data["score"] = max(0, min(100, int(data.get("score", 70))))
        return data, None

    except Exception as e:
        return None, f"OpenAI error: {str(e)[:180]}"

def local_synthesis(gemini, openai):
    # Emergency synthesis so Deep Validation can still show a useful result
    # if the final Gemini synthesis call fails.
    return {
        "consensus": "Both AI perspectives were generated independently. Treat the shared points as hypotheses to validate with real customers rather than verified market facts.",
        "common_strengths": [
            gemini.get("strength", ""),
            openai.get("strength", "")
        ][:2],
        "common_concerns": [
            gemini.get("weakness", ""),
            openai.get("risks", "")
        ][:2],
        "key_disagreement": "The two AI perspectives may differ in how strongly they interpret demand, pricing and execution risk. Validate these points with real customer evidence.",
        "decision_focus": [
            "Customer willingness to pay",
            "Unit economics and operating cost",
            "Competitor alternatives and differentiation"
        ],
        "final_action_plan": [
            "Interview 10-15 target customers",
            "Test two or three price points with a small pilot",
            "Measure demand, cost and repeat usage before scaling"
        ],
        "final_recommendation": "Use the AI outputs as a preliminary validation map, not as proof of market demand. Run a small real-world pilot and use the results to decide what to change before investing further."
    }

def synthesize(gemini, openai):
    key = env("GEMINI_API_KEY")
    if not key:
        return local_synthesis(gemini, openai), None

    prompt = (
        "Business idea validation synthesis. Compare these two independent analyses.\n\n"
        "GEMINI ANALYSIS:\n" + json.dumps(gemini, ensure_ascii=False) +
        "\n\nOPENAI ANALYSIS:\n" + json.dumps(openai, ensure_ascii=False)
    )

    body = {
        "system_instruction": {"parts": [{"text": DEEP_SYNTHESIS_SYSTEM}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": key},
            json=body,
            timeout=45
        )
        if not r.ok:
            return local_synthesis(gemini, openai), "Final Gemini synthesis failed: " + api_error(r)

        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return clean_json(raw), None

    except Exception as e:
        return local_synthesis(gemini, openai), f"Final synthesis fallback used: {str(e)[:160]}"

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/api/health")
def health():
    # Never returns the actual secrets.
    return jsonify({
        "status": "ok",
        "gemini_configured": bool(env("GEMINI_API_KEY")),
        "openai_configured": bool(env("OPENAI_API_KEY")),
        "openai_model": env("OPENAI_MODEL") or "gpt-5.6-luna"
    })

@app.post("/api/analyze")
def analyze():
    d = request.get_json(silent=True) or {}

    if not d.get("idea"):
        return jsonify({"error": "Business idea is required"}), 400

    mode = d.get("mode", "quick")

    if mode != "deep":
        data, err = gemini_analysis(d)
        if data is None:
            data = fallback(d)
            data["_provider"] = "Fallback"
            data["_notice"] = err
        else:
            data["_provider"] = "Gemini"
        return jsonify(data)

    gemini_key_present = bool(env("GEMINI_API_KEY"))
    openai_key_present = bool(env("OPENAI_API_KEY"))

    if not gemini_key_present or not openai_key_present:
        return jsonify({
            "error": "Deep Validation cannot see both API keys in the running Render process.",
            "gemini_configured": gemini_key_present,
            "openai_configured": openai_key_present,
            "next_step": "Open /api/health after the latest deploy. It shows only true/false, never the keys."
        }), 503

    g, gerr = gemini_analysis(d)
    if g is None:
        return jsonify({"error": "Gemini analysis failed", "detail": gerr}), 502

    o, oerr = openai_analysis(d)
    if o is None:
        return jsonify({"error": "OpenAI analysis failed", "detail": oerr}), 502

    synthesis, serr = synthesize(g, o)

    result = {
        "mode": "deep",
        "gemini": g,
        "openai": o,
        "synthesis": synthesis,
        "_providers": ["Gemini", "OpenAI"]
    }

    if serr:
        result["_notice"] = serr

    return jsonify(result)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
