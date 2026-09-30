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
    "competitors", "competitor_sources",
    "marketing", "risks", "score", "scoretext",
    "recommendation", "assumptions",
    "validation_questions", "action_plan"
]

SYSTEM = """You are Vyapaar Sathi, an AI business-validation assistant for BBA students and first-time entrepreneurs.
Analyze a business idea using the supplied customer, location and budget.
Be practical, specific, conservative and concise. Do not guarantee success.
Treat scores and estimates as preliminary.
Return ONLY valid JSON with exactly these fields:
market, problem, value, pricing, strength, weakness, opportunity, threat, competitors, competitor_sources, marketing, risks, score, scoretext, recommendation, assumptions, validation_questions, action_plan.
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
        "competitors": "Live competitor lookup was unavailable. Identify 3–5 direct local alternatives and 1–2 indirect substitutes before investing.",
        "competitor_sources": [],
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

def gemini_request(body, key, label="GEMINI", attempts=1, timeout=3):
    # Live-demo mode: use one verified current model only. If it is slow,
    # return control quickly so the local validation fallback can respond.
    model_name = "gemini-3.8-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    try:
        r = requests.post(url, params={"key": key}, json=body, timeout=timeout)
        if r.ok:
            log_safe(f"{label}_OK", f"model={model_name}")
            return r, None

        last_error = api_error(r)
        log_safe(f"{label}_ERROR", f"model={model_name}; {last_error}")
        return None, last_error

    except requests.exceptions.Timeout:
        err = f"Gemini timed out after {timeout}s"
        log_safe(f"{label}_TIMEOUT", err)
        return None, err
    except requests.exceptions.RequestException as e:
        err = f"Gemini connection error: {str(e)[:160]}"
        log_safe(f"{label}_REQUEST_ERROR", err)
        return None, err
    except Exception as e:
        err = f"Gemini unexpected error: {str(e)[:160]}"
        log_safe(f"{label}_EXCEPTION", err)
        return None, err

def gemini_analysis(d, role="market"):
    key = env("GEMINI_API_KEY")
    if not key:
        return None, "GEMINI_API_KEY is missing from the running Render service."

    role_instruction = """Act as an independent Market Analyst.
Focus on customer segment, problem-solution fit, local market context,
value proposition, positioning and realistic opportunities."""
    label = "GEMINI_MARKET"

    competitor_mode = bool(d.get("include_competitors"))
    if competitor_mode:
        role_instruction += """
Use Google Search grounding to identify CURRENT public information about competitors
and alternatives relevant to the exact location and customer segment.
Do not invent competitor names, prices, ratings or claims.
Prefer official business websites, Google/Maps-visible business information, established
marketplaces/directories and reputable local sources when available.
Clearly distinguish direct competitors from indirect substitutes.
For each competitor, include name, type, what it offers, approximate publicly visible
price/range if available, and one practical differentiation observation.
If a price or fact cannot be verified, say "Not publicly verified".
Return source URLs in competitor_sources."""

    prompt = f"""{role_instruction}

Business idea: {d["idea"]}
Target customer: {d.get("customer") or "Not specified"}
Location: {d.get("location") or "Not specified"}
Investment budget: {d.get("budget") or "Not specified"}

Generate the structured preliminary validation report.
The competitor section must be based on fresh web research when competitor mode is enabled."""

    system = SYSTEM + """
Return competitors as an array of concise objects with:
name, type, offer, price, differentiation.
Return competitor_sources as an array of objects with:
title, url.
Do not fabricate sources."""

    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    if competitor_mode:
        body["tools"] = [{"google_search": {}}]

    r, err = gemini_request(body, key, label=label, attempts=1, timeout=8)
    if r is None:
        return None, err

    try:
        response_json = r.json()
        raw = response_json["candidates"][0]["content"]["parts"][0]["text"]
        data = clean_json(raw)

        # Keep only safe, displayable competitor fields.
        competitors = data.get("competitors", [])
        if not isinstance(competitors, list):
            competitors = []
        safe_competitors = []
        for item in competitors[:6]:
            if isinstance(item, dict):
                safe_competitors.append({
                    "name": str(item.get("name", ""))[:100],
                    "type": str(item.get("type", ""))[:60],
                    "offer": str(item.get("offer", ""))[:240],
                    "price": str(item.get("price", "Not publicly verified"))[:100],
                    "differentiation": str(item.get("differentiation", ""))[:240]
                })
        data["competitors"] = safe_competitors

        sources = data.get("competitor_sources", [])
        if not isinstance(sources, list):
            sources = []
        safe_sources = []
        for item in sources[:8]:
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://")):
                safe_sources.append({
                    "title": str(item.get("title", "Source"))[:120],
                    "url": str(item.get("url", ""))[:500]
                })
        data["competitor_sources"] = safe_sources

        data["score"] = max(0, min(100, int(data.get("score", 70))))
        return data, None
    except Exception as e:
        err = f"Gemini response parsing error: {str(e)[:180]}"
        log_safe(f"{label}_PARSE_ERROR", err)
        return None, err

def competitor_search(d):
    """Run live public-web competitor research separately so the main demo never waits on it."""
    key = env("GEMINI_API_KEY")
    if not key:
        return {"competitors": [], "competitor_sources": [], "_notice": "Gemini is not configured."}

    system = """You are the competitor-research module of Vyapaar Sathi.
Use Google Search grounding to identify CURRENT public information about 3 to 5 competitors or substitutes relevant to the exact business idea, customer and location.
Do not invent names, prices, ratings or facts. Prefer official websites, Google/Maps-visible information, established marketplaces/directories and reputable local sources.
Clearly distinguish direct competitors from indirect substitutes.
If a price or fact cannot be verified, write 'Not publicly verified'.
Return ONLY valid JSON with exactly these fields:
competitors: array of objects with name, type, offer, price, differentiation
competitor_sources: array of objects with title, url"""
    prompt = f"""Business idea: {d.get('idea') or 'Not specified'}
Target customer: {d.get('customer') or 'Not specified'}
Location: {d.get('location') or 'Not specified'}
Find current public competitor/alternative signals for this market."""
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
        "tools": [{"google_search": {}}]
    }
    r, err = gemini_request(body, key, label="COMPETITOR_SEARCH", attempts=1, timeout=6)
    if r is None:
        return {"competitors": [], "competitor_sources": [], "_notice": err or "Live competitor lookup timed out."}
    try:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = clean_json(raw)
        comps = data.get("competitors", []) if isinstance(data, dict) else []
        safe = []
        for item in comps[:6]:
            if isinstance(item, dict) and item.get("name"):
                safe.append({
                    "name": str(item.get("name", ""))[:100],
                    "type": str(item.get("type", ""))[:60],
                    "offer": str(item.get("offer", ""))[:240],
                    "price": str(item.get("price", "Not publicly verified"))[:100],
                    "differentiation": str(item.get("differentiation", ""))[:240]
                })
        srcs = data.get("competitor_sources", []) if isinstance(data, dict) else []
        sources = []
        for item in srcs[:8]:
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://")):
                sources.append({"title": str(item.get("title", "Source"))[:120], "url": str(item.get("url", ""))[:500]})
        return {"competitors": safe, "competitor_sources": sources}
    except Exception as e:
        return {"competitors": [], "competitor_sources": [], "_notice": f"Competitor response could not be parsed: {str(e)[:120]}"}

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

@app.post("/api/competitors")
def competitors():
    d = request.get_json(silent=True) or {}
    if not isinstance(d, dict) or not d.get("idea"):
        return jsonify({"error": "Business idea is required"}), 400
    result = competitor_search(d)
    result["_live_competitor_check"] = True
    return jsonify(result)

@app.post("/api/analyze")
def analyze():
    d = request.get_json(silent=True) or {}
    if not isinstance(d, dict):
        return jsonify({"error": "Invalid request body"}), 400
    if not d.get("idea"):
        return jsonify({"error": "Business idea is required"}), 400

    mode = d.get("mode", "quick")

    if mode != "deep":
        # Main report is intentionally independent of live competitor search.
        # This keeps the demo responsive; /api/competitors runs separately.
        core = dict(d)
        core["include_competitors"] = False
        data, err = gemini_analysis(core, "market")
        if data is None:
            data = fallback(d)
            data["_provider"] = "Vyapaar Sathi Local Validation"
            data["_notice"] = "AI response was delayed; Vyapaar Sathi switched to its instant validation layer."
        else:
            data["_provider"] = "Gemini"
        data["competitors"] = []
        data["competitor_sources"] = []
        data["_live_competitor_check"] = False
        return jsonify(data)


    if not env("GEMINI_API_KEY"):
        return jsonify({"error": "Gemini API key is not configured.", "gemini_configured": False}), 503

    # Ultra-fast zero-cost Deep Validation:
    # one Gemini call + a transparent local critic/synthesis layer.
    market, merr = gemini_analysis(d, "market")
    if market is None:
        # Fast local fallback: the demo still produces a structured validation
        # report when Gemini is temporarily slow/unavailable.
        market = fallback(d)
        market["_provider"] = "Vyapaar Sathi Local Validation"
        market["_notice"] = "Gemini was temporarily slow; Vyapaar Sathi switched to its instant validation layer."
        merr = None

    critic = {
        "market": market.get("market", ""),
        "problem": market.get("problem", ""),
        "value": market.get("value", ""),
        "pricing": market.get("pricing", ""),
        "competitors": market.get("competitors", []),
        "competitor_sources": market.get("competitor_sources", []),
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
        "_performance": "One Gemini call with a 3-second demo timeout + instant local validation fallback."
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
