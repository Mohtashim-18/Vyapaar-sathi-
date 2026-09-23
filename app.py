import os,json
from flask import Flask,request,jsonify,send_from_directory
import requests

app=Flask(__name__,static_folder=".",static_url_path="")
SYSTEM="""You are Vyapaar Sathi, an AI business-analysis assistant for BBA students.
Analyze any business idea using the supplied customer, location and budget.
Be practical, concise and specific to the idea. Do not guarantee success.
Return ONLY valid JSON with exactly:
market, problem, value, pricing, strength, weakness, opportunity, threat, marketing, risks, score, scoretext, recommendation.
score is an illustrative preliminary assessment from 0 to 100, not a forecast."""

def fallback(d):
    idea=d["idea"]; c=d.get("customer") or "the target customer"; loc=d.get("location") or "the target location"; b=d.get("budget") or "the stated budget"
    return {"market":f"{c} in {loc}, with demand linked to the need described in '{idea}'.",
    "problem":"The business should solve a specific customer pain point better, cheaper, faster or more conveniently than alternatives.",
    "value":f"A focused offering designed around the core customer need in '{idea}'.",
    "pricing":f"Test multiple price points with real customers. Budget context: {b}. Possible revenue can include one-time sales, subscriptions, commissions or add-ons depending on the model.",
    "strength":"Clear positioning and a focused customer segment can simplify early marketing.",
    "weakness":"Demand, operations and willingness-to-pay are still unvalidated.",
    "opportunity":"Local partnerships, digital acquisition and niche targeting can support a pilot.",
    "threat":"Existing competitors, price pressure and changing customer preferences.",
    "marketing":"Use local digital channels, referrals, partnerships and a small pilot campaign.",
    "risks":"Weak demand, wrong pricing, operating costs and customer retention.",
    "score":70,"scoretext":"Fallback prototype assessment. Connect Gemini for idea-specific AI analysis.",
    "recommendation":"Interview potential customers, study competitors, test pricing and run a small pilot before scaling."}

@app.get("/")
def home(): return send_from_directory(".","index.html")

@app.post("/api/analyze")
def analyze():
    d=request.get_json(silent=True) or {}
    if not d.get("idea"): return jsonify({"error":"Business idea is required"}),400
    key=os.getenv("GEMINI_API_KEY","").strip()
    if not key:return jsonify(fallback(d))
    prompt=f"""Business idea: {d['idea']}
Target customer: {d.get('customer') or 'Not specified'}
Location: {d.get('location') or 'Not specified'}
Investment budget: {d.get('budget') or 'Not specified'}
Generate the JSON report."""
    url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    try:
      r=requests.post(url,params={"key":key},json={"system_instruction":{"parts":[{"text":SYSTEM}]},"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json","temperature":0.35}},timeout=45)
      r.raise_for_status(); raw=r.json()["candidates"][0]["content"]["parts"][0]["text"]; data=json.loads(raw)
      data["score"]=max(0,min(100,int(data.get("score",70))));return jsonify(data)
    except Exception as e:return jsonify({"error":"AI API request failed. Check the Gemini API key and internet connection.","detail":str(e)[:160]}),502

if __name__=="__main__":
    port=int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0",port=port,debug=False)
