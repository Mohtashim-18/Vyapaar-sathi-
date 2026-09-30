import os, json, math, requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

MODEL = 'gemini-3.8-flash'

SYSTEM = '''You are Vyapaar Sathi, a practical business validation engine for first-time entrepreneurs.
Return only valid JSON. Be specific, conservative and evidence-aware. Never guarantee success.
Fields: market, problem, value, pricing, strength, weakness, opportunity, threat, marketing, risks,
score, scoretext, recommendation, assumptions, validation_questions, action_plan.
Arrays must contain exactly 3 concise strings. Score is 0-100 and is a preliminary hypothesis, not market proof.'''


def env(k):
    v = os.environ.get(k, '')
    return v.strip() if isinstance(v, str) else ''


def clean_json(text):
    text = (text or '').strip()
    if text.startswith('```'):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == '```': lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return json.loads(text)


def gemini(body, timeout=5):
    key = env('GEMINI_API_KEY')
    if not key: return None, 'Gemini API key is not configured.'
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'
    try:
        r = requests.post(url, params={'key': key}, json=body, timeout=timeout)
        if not r.ok:
            try: msg = r.json().get('error', {}).get('message', r.text[:180])
            except Exception: msg = r.text[:180]
            return None, f'HTTP {r.status_code}: {msg}'
        raw = r.json()['candidates'][0]['content']['parts'][0]['text']
        return clean_json(raw), None
    except requests.exceptions.Timeout:
        return None, f'AI response timed out after {timeout}s.'
    except Exception as e:
        return None, str(e)[:220]


def fallback(d):
    idea, customer, loc, budget = d.get('idea',''), d.get('customer','target customers'), d.get('location','target location'), d.get('budget','stated budget')
    return {
        'market': f'{customer} in {loc} are the first segment to test for “{idea}”. Demand should be measured through real responses, orders or sign-ups.',
        'problem': 'The idea needs a clearly defined pain point that is frequent enough for the customer to change behaviour.',
        'value': 'A focused offer should be easier to understand, test and improve than a broad “everyone” proposition.',
        'pricing': f'Test 2–3 price points against alternatives. Current budget context: {budget}.',
        'strength': 'Clear target customer and a small pilot make the idea relatively easy to test.',
        'weakness': 'Willingness-to-pay, repeat demand and unit economics are not yet proven.',
        'opportunity': 'Local partnerships, referrals and a focused first neighbourhood/campus can reduce early acquisition cost.',
        'threat': 'Existing alternatives, price sensitivity and inconsistent service quality can slow adoption.',
        'marketing': 'Start with one reachable channel, one offer and one measurable call-to-action.',
        'risks': 'Do not spend the full budget before measuring conversion, margin, repeat usage and complaints.',
        'score': 72, 'scoretext': 'Pilot-ready hypothesis; evidence from real customers is still required.',
        'recommendation': 'Run a 7–14 day paid pilot, test pricing and track unit economics before scaling.',
        'assumptions': ['The target customer experiences the stated problem', 'Customers will pay the tested price', 'Delivery can be profitable at small scale'],
        'validation_questions': ['What do customers use today instead?', 'What price would they actually pay?', 'What result would make them return or recommend it?'],
        'action_plan': ['Interview 10–15 target customers', 'Run a small paid pilot with 2 price points', 'Track conversion, contribution margin and repeat usage']
    }


def safe_ai(d):
    prompt = f'''{SYSTEM}\n\nBusiness idea: {d.get('idea')}\nTarget customer: {d.get('customer')}\nLocation: {d.get('location')}\nBudget: {d.get('budget')}\n\nProduce a structured preliminary validation report.''' 
    body = {'system_instruction': {'parts':[{'text':SYSTEM}]}, 'contents':[{'parts':[{'text':prompt}]}], 'generationConfig':{'responseMimeType':'application/json'}}
    data, err = gemini(body, 5)
    return data if data else fallback(d), err

# Evidence snapshots used only to make the college-demo scenario concrete.
# They are public-web observations and are explicitly labelled as snapshots.
TIFFIN_SNAPSHOT = [
    {'name':'Apna Tiffin', 'type':'Direct', 'offer':'Student-focused home-style tiffin; hostel/PG delivery in Lucknow', 'price':'₹95+; lunch ₹115; L+D ₹195/day', 'gap':'Student scheduling, skip-day flexibility and campus/hostel focus', 'url':'https://apnatiffin.org/tiffin-for-college-students-lucknow'},
    {'name':'Lucknowi Tiffin', 'type':'Direct / platform', 'offer':'Homemade tiffin platform with packages and online ordering', 'price':'30 meals ₹3,450; 10 meals ₹1,200', 'gap':'Online ordering and multiple package choices', 'url':'https://www.lucknowitiffin.in/orders/178'},
    {'name':'AF Tiffin Service', 'type':'Direct / local', 'offer':'Student and general tiffin delivery in Lucknow', 'price':'Student tiffin advertised from ₹50 after discount', 'gap':'Low-price student positioning', 'url':'https://sites.google.com/view/af-tiffin-service/1'},
    {'name':'Tifola', 'type':'Indirect / marketplace', 'offer':'Monthly tiffin marketplace connecting customers with local kitchens', 'price':'Vendor-specific live pricing', 'gap':'Multiple kitchens and pause control', 'url':'https://tifola.com/tiffin-service/lucknow/category/monthly-tiffin'}
]


def scan_competitors(d):
    idea = (d.get('idea') or '').lower()
    if any(x in idea for x in ['tiffin','meal','lunch','dinner','food service']):
        return {'mode':'public_snapshot','updated':'30 Sep 2026','competitors':TIFFIN_SNAPSHOT,
                'notice':'Demo market snapshot based on public webpages. Prices and availability can change; verify before use.'}
    # Generic alternatives for other ideas — intentionally no fabricated company names.
    return {'mode':'framework','updated':'live lookup unavailable in demo mode','competitors':[
        {'name':'Local direct competitor', 'type':'Direct', 'offer':'Same customer + same core problem', 'price':'Verify locally', 'gap':'Compare price, speed, quality and convenience', 'url':''},
        {'name':'Online / marketplace alternative', 'type':'Indirect', 'offer':'Digital or platform-based substitute', 'price':'Verify current listing', 'gap':'Compare reach, fees and customer trust', 'url':''},
        {'name':'DIY / status quo', 'type':'Substitute', 'offer':'Customer solves the problem themselves', 'price':'Time + existing spend', 'gap':'Your offer must be meaningfully easier or better', 'url':''}
    ], 'notice':'For this idea, verify named local competitors before making a market claim.'}

@app.get('/')
def home(): return send_from_directory('.', 'index.html')

@app.get('/api/health')
def health(): return jsonify({'status':'ok','gemini_configured':bool(env('GEMINI_API_KEY')),'mode':'Standout validation prototype'})

@app.post('/api/analyze')
def analyze():
    d=request.get_json(silent=True) or {}
    if not d.get('idea'): return jsonify({'error':'Business idea is required'}),400
    data, err=safe_ai(d)
    data['score']=max(0,min(100,int(data.get('score',72))))
    data['_provider']='Gemini' if err is None else 'Vyapaar Sathi instant validation layer'
    data['_notice']=err if err else ''
    return jsonify(data)

@app.post('/api/competitors')
def competitors():
    d=request.get_json(silent=True) or {}
    if not d.get('idea'): return jsonify({'error':'Business idea is required'}),400
    return jsonify(scan_competitors(d))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5000')),debug=False)
