import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor
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

def env(name):
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

def log_safe(label, message):
    print(f"[Vyapaar Sathi] {label}: {message}", flush=True)

def gemini_request(body, key, label="GEMINI", attempts=1, timeout=8):
    # Fallback order keeps the free demo resilient to temporary model load.
    model_candidates = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite"
    ]
    last_error = "Unknown Gemini error"

    for model_name in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(url, params={"key": key}, json=body, timeout=timeout)
                if r.ok:
                    log_safe(f"{label}_OK", f"model={model_name}")
                    return r, None

                last_error = api_error(r)
                retryable = r.status_code in (408, 429) or 500 <= r.status_code <= 599
                if not retryable:
                    log_safe(f"{label}_ERROR", f"model={model_name}; {last_error}")
                    break

                if attempt < attempts:
                    delay = 2 ** (attempt - 1) + 0.5
                    log_safe(f"{label}_RETRY", f"model={model_name}; retry {attempt}/{attempts}; {delay:.1f}s")
                    time.sleep(delay)
                else:
                    log_safe(f"{label}_MODEL_FALLBACK", f"model={model_name}; trying next model")

            except requests.exceptions.Timeout as e:
                last_error = f"Gemini timeout after {timeout}s: {str(e)[:120]}"
                if attempt < attempts:
                    delay = 2 ** (attempt - 1) + 0.5
                    log_safe(f"{label}_RETRY", f"model={model_name}; timeout; {delay:.1f}s")
                    time.sleep(delay)
                else:
                    log_safe(f"{label}_MODEL_FALLBACK", f"model={model_name}; timeout; trying next model")
            except requests.exceptions.RequestException as e:
                last_error = f"Gemini network error: {str(e)[:160]}"
                log_safe(f"{label}_MODEL_FALLBACK", f"model={model_name}; {last_error}")
                break

    log_safe(f"{label}_ERROR", last_error)
    return None, last_error

def gemini_analysis(d, role="market"):
    key = env("GEMINI_API_KEY")
    if not key:
        return None, "GEMINI_API_KEY is missing from the running Render service."

    if role == "critic":
        role_instruction = """Act as an independent skeptical Business Critic.
Focus especially on customer willingness-to-pay, competition, pricing, unit economics,
execution risks and what evidence must be collected before investing.
Do not simply agree with another analyst; form your own view."""
        label = "GEMINI_CRITIC"
    else:
        role_instruction = """Act as an independent Market Analyst.
Focus especially on customer segment, problem-solution fit, local market context,
value proposition, positioning and realistic opportunities."""
        label = "GEMINI_MARKET"

    prompt = f"""{role_instruction}

Business idea: {d["idea"]}
Target customer: {d.get("customer") or "Not specified"}
Location: {d.get("location") or "Not specified"}
Investment budget: {d.get("budget") or "Not specified"}

Generate the structured preliminary validation report independently."""

    system = SYSTEM + "\n\nYour current role is independent and must not assume any other AI has already analyzed this idea."

    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    r, err = gemini_request(
        body, key, label=label,
        attempts=1, timeout=15
    )
    if r is None:
        return None, err

    try:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = clean_json(raw)
        data["score"] = max(0, min(100, int(data.get("score", 70))))
        return data, None
    except Exception as e:
        err = f"Gemini response parsing error: {str(e)[:180]}"
        log_safe(f"{label}_PARSE_ERROR", err)
        return None, err

def local_synthesis(gemini, critic):
    def val(obj, key, default=""):
        return obj.get(key, default) if isinstance(obj, dict) else default

    return {
        "consensus": (
            "Both Gemini perspectives independently point to the same core principle: "
            "the idea should be tested with real customers before significant investment. "
            "Their outputs are hypotheses, not verified market data."
        ),
        "common_strengths": [
            val(gemini, "strength", "Focused customer positioning can help."),
            val(critic, "strength", "A clear problem-solution fit can help.")
        ],
        "common_concerns": [
            val(gemini, "weakness", "Demand still needs validation."),
            val(critic, "risks", "Pricing, costs and competition need evidence.")
        ],
        "key_disagreement": (
            "The two independent perspectives may differ in how strongly they interpret "
            "market opportunity and execution risk. Use customer interviews, competitor "
            "checks and a small paid pilot to resolve the disagreement."
        ),
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
        "final_recommendation": (
            "Use the two independent Gemini perspectives as a preliminary validation map, "
            "not as proof of market demand. Run a small real-world pilot and use the results "
            "to refine the idea before committing more capital."
        )
    }

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    log_safe("UNHANDLED_ERROR", str(e)[:240])
    return jsonify({
        "error": "Server error while processing the request",
        "detail": str(e)[:240]
    }), 500

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gemini_configured": bool(env("GEMINI_API_KEY")),
        "openai_required": False,
        "mode": "Gemini-only zero-cost Deep Validation"
    })

@app.get("/api/test-gemini")
def test_gemini():
    key = env("GEMINI_API_KEY")
    if not key:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY is missing"}), 503

    body = {
        "contents": [{"parts": [{"text": "Reply with exactly: GEMINI_OK"}]}],
        "generationConfig": {}
    }

    r, err = gemini_request(body, key, label="GEMINI_TEST", attempts=1, timeout=20)
    if r is None:
        return jsonify({"ok": False, "error": err}), 502

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"ok": True, "response": text[:80]})
    except Exception as e:
        err = f"Gemini response parsing error: {str(e)[:180]}"
        return jsonify({"ok": False, "error": err}), 502

@app.post("/api/analyze")
def analyze():
    d = request.get_json(silent=True) or {}
    if not isinstance(d, dict):
        return jsonify({"error": "Invalid request body"}), 400
    if not d.get("idea"):
        return jsonify({"error": "Business idea is required"}), 400

    mode = d.get("mode", "quick")

    if mode != "deep":
        data, err = gemini_analysis(d, "market")
        if data is None:
            data = fallback(d)
            data["_provider"] = "Fallback"
            data["_notice"] = err
        else:
            data["_provider"] = "Gemini"
        return jsonify(data)

    if not env("GEMINI_API_KEY"):
        return jsonify({"error": "Gemini API key is not configured.", "gemini_configured": False}), 503

    # Ultra-fast zero-cost Deep Validation:
    # one Gemini call + a transparent local critic/synthesis layer.
    market, merr = gemini_analysis(d, "market")
    if market is None:
        return jsonify({"error": "Gemini analysis failed", "detail": merr}), 502

    critic = {
        "market": market.get("market", ""),
        "problem": market.get("problem", ""),
        "value": market.get("value", ""),
        "pricing": market.get("pricing", ""),
        "strength": "Defined customer segment and a clear problem-solution direction.",
        "weakness": "Willingness-to-pay and repeat demand are not yet verified.",
        "opportunity": "A small local pilot can test demand, pricing and repeat usage.",
        "threat": "Existing alternatives, price competition and operating-cost pressure.",
        "marketing": market.get("marketing", ""),
        "risks": market.get("risks", ""),
        "score": market.get("score", 70),
        "scoretext": "Preliminary AI score; validate with real customer evidence.",
        "recommendation": "Run a small paid pilot, compare competitors and measure unit economics before scaling.",
        "assumptions": [
            "The stated customer segment has the problem",
            "Customers will pay the tested price",
            "The offering can be delivered at sustainable cost"
        ],
        "validation_questions": [
            "How do customers solve this problem today?",
            "What price would they actually pay?",
            "Would they use the product or service repeatedly?"
        ],
        "action_plan": [
            "Interview 10-15 target customers",
            "Run a small paid pilot",
            "Track demand, cost, conversion and repeat usage"
        ]
    }

    synthesis = {
        "consensus": (
            "The AI analysis and structured business-critic layer agree that the idea "
            "has a testable customer problem, but real demand and willingness-to-pay "
            "must be verified before significant investment."
        ),
        "common_strengths": [
            market.get("strength", "Clear customer focus"),
            "The idea can be tested through a small local pilot."
        ],
        "common_concerns": [
            market.get("weakness", "Demand still needs validation."),
            "Pricing, operating costs and repeat demand need real-world evidence."
        ],
        "key_disagreement": (
            "The AI-generated market opportunity is a preliminary hypothesis, while "
            "the critic layer takes a conservative view until customer evidence exists."
        ),
        "decision_focus": [
            "Customer willingness to pay",
            "Unit economics and operating cost",
            "Competitor alternatives and differentiation"
        ],
        "final_action_plan": [
            "Interview 10-15 target customers",
            "Test 2-3 price points with a small paid pilot",
            "Measure demand, cost and repeat usage before scaling"
        ],
        "final_recommendation": (
            "Proceed to a small validation pilot rather than investing the full budget. "
            "Use real customer responses and unit economics to decide the next step."
        )
    }

    return jsonify({
        "mode": "deep",
        "gemini": market,
        "critic": critic,
        "synthesis": synthesis,
        "_providers": ["Gemini AI Analyst", "Vyapaar Sathi Business Critic"],
        "_synthesis": "Vyapaar Sathi structured validation layer",
        "_cost_mode": "zero-cost API path; no OpenAI credits required",
        "_performance": "Single Gemini call + local validation/synthesis for fast live demos."
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
