import os, json, requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

ANALYSIS_FIELDS = [
    'market','problem','value','pricing','strength','weakness','opportunity','threat',
    'marketing','risks','score','scoretext','recommendation','assumptions','validation_questions','action_plan'
]

SYSTEM = '''You are Vyapaar Sathi, an AI business-validation assistant for BBA students and first-time entrepreneurs.
Analyze a business idea using the supplied customer, location and budget.
Be practical, specific, conservative and concise. Do not guarantee success. Treat all scores and estimates as preliminary.
Return ONLY valid JSON with exactly these fields:
market, problem, value, pricing, strength, weakness, opportunity, threat, marketing, risks, score, scoretext, recommendation, assumptions, validation_questions, action_plan.
score must be an integer from 0 to 100 and is an illustrative preliminary assessment, not a forecast.
assumptions, validation_questions and action_plan must each be arrays of 3 concise strings.'''

DEEP_SYNTHESIS_SYSTEM = '''You are the final synthesis engine for Vyapaar Sathi.
You receive two independent preliminary business analyses: one from Gemini and one from OpenAI.
Compare them without pretending either is verified market data. Produce a neutral, evidence-aware synthesis.
Return ONLY valid JSON with exactly:
consensus, common_strengths, common_concerns, key_disagreement, decision_focus, final_action_plan, final_recommendation.
All list fields must contain 2-4 concise strings. final_recommendation must be 2-4 sentences.'''

def fallback(d):
    idea=d['idea']; c=d.get('customer') or 'the target customer'; loc=d.get('location') or 'the target location'; b=d.get('budget') or 'the stated budget'
    return {
        'market':f'{c} in {loc}, with demand linked to the need described in "{idea}".',
        'problem':'The business should solve a specific customer pain point better, cheaper, faster or more conveniently than alternatives.',
        'value':f'A focused offering designed around the core customer need in "{idea}".',
        'pricing':f'Test multiple price points with real customers. Budget context: {b}.',
        'strength':'Clear positioning and a focused customer segment can simplify early marketing.',
        'weakness':'Demand, operating costs and willingness-to-pay are still unvalidated.',
        'opportunity':'Local partnerships, referrals and a small pilot can create useful early evidence.',
        'threat':'Existing alternatives, price pressure and changing customer preferences.',
        'marketing':'Use local digital channels, referrals, partnerships and a small pilot campaign.',
        'risks':'Weak demand, wrong pricing, operating costs and customer retention.',
        'score':70,'scoretext':'Fallback prototype assessment. Connect Gemini for idea-specific AI analysis.',
        'recommendation':'Interview potential customers, study alternatives, test pricing and run a small pilot before scaling.',
        'assumptions':['Customers have the stated problem','Customers will pay the tested price','The business can deliver at sustainable cost'],
        'validation_questions':['How do you solve this problem today?','What would make you switch?','What price would feel reasonable?'],
        'action_plan':['Interview 10 target customers','Run a small paid pilot','Compare results against cost and retention']
    }

def clean_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1] if '\n' in text else text
        if text.endswith('```'): text=text[:-3]
    return json.loads(text.strip())

def gemini_analysis(d):
    key=os.getenv('GEMINI_API_KEY','').strip()
    if not key: return None, 'GEMINI_API_KEY is not configured'
    prompt=f'''Business idea: {d['idea']}
Target customer: {d.get('customer') or 'Not specified'}
Location: {d.get('location') or 'Not specified'}
Investment budget: {d.get('budget') or 'Not specified'}
Generate the structured preliminary validation report.'''
    url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
    body={'system_instruction':{'parts':[{'text':SYSTEM}]},'contents':[{'parts':[{'text':prompt}]}],
          'generationConfig':{'responseMimeType':'application/json','temperature':0.25}}
    try:
        r=requests.post(url,params={'key':key},json=body,timeout=45); r.raise_for_status()
        raw=r.json()['candidates'][0]['content']['parts'][0]['text']
        data=clean_json(raw); data['score']=max(0,min(100,int(data.get('score',70))))
        return data,None
    except Exception as e: return None, str(e)[:180]

def openai_analysis(d):
    key=os.getenv('OPENAI_API_KEY','').strip()
    if not key: return None, 'OPENAI_API_KEY is not configured'
    model=os.getenv('OPENAI_MODEL','gpt-5.6-luna').strip()
    prompt=f'''Analyze this business idea independently as a skeptical business critic.
Business idea: {d['idea']}
Target customer: {d.get('customer') or 'Not specified'}
Location: {d.get('location') or 'Not specified'}
Investment budget: {d.get('budget') or 'Not specified'}
Return ONLY JSON with fields: market, problem, value, pricing, strength, weakness, opportunity, threat, marketing, risks, score, scoretext, recommendation, assumptions, validation_questions, action_plan.
score is illustrative from 0-100. arrays assumptions, validation_questions and action_plan must each have 3 concise strings.'''
    schema={"type":"json_schema","name":"business_analysis","strict":True,"schema":{"type":"object","properties":{
        'market':{'type':'string'},'problem':{'type':'string'},'value':{'type':'string'},'pricing':{'type':'string'},
        'strength':{'type':'string'},'weakness':{'type':'string'},'opportunity':{'type':'string'},'threat':{'type':'string'},
        'marketing':{'type':'string'},'risks':{'type':'string'},'score':{'type':'integer'},'scoretext':{'type':'string'},
        'recommendation':{'type':'string'},'assumptions':{'type':'array','items':{'type':'string'}},
        'validation_questions':{'type':'array','items':{'type':'string'}},'action_plan':{'type':'array','items':{'type':'string'}}
    },'required':ANALYSIS_FIELDS,'additionalProperties':False}}
    try:
        r=requests.post('https://api.openai.com/v1/responses',headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},
                        json={'model':model,'input':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                              'text':{'format':schema}},timeout=60)
        r.raise_for_status(); j=r.json()
        raw=j.get('output_text')
        if not raw:
            for item in j.get('output',[]):
                for part in item.get('content',[]):
                    if part.get('type') in ('output_text','text') and part.get('text'): raw=part['text']; break
                if raw: break
        data=clean_json(raw); data['score']=max(0,min(100,int(data.get('score',70))))
        return data,None
    except Exception as e: return None, str(e)[:180]

def synthesize(gemini, openai):
    key=os.getenv('GEMINI_API_KEY','').strip()
    if not key: return None, 'Gemini key required for synthesis'
    prompt='''Business idea validation synthesis. Compare these two independent analyses.\n\nGEMINI ANALYSIS:\n'''+json.dumps(gemini,ensure_ascii=False)+'''\n\nOPENAI ANALYSIS:\n'''+json.dumps(openai,ensure_ascii=False)
    body={'system_instruction':{'parts':[{'text':DEEP_SYNTHESIS_SYSTEM}]},'contents':[{'parts':[{'text':prompt}]}],
          'generationConfig':{'responseMimeType':'application/json','temperature':0.2}}
    try:
        r=requests.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',params={'key':key},json=body,timeout=45); r.raise_for_status()
        raw=r.json()['candidates'][0]['content']['parts'][0]['text']; return clean_json(raw),None
    except Exception as e: return None,str(e)[:180]

@app.get('/')
def home(): return send_from_directory('.', 'index.html')

@app.post('/api/analyze')
def analyze():
    d=request.get_json(silent=True) or {}
    if not d.get('idea'): return jsonify({'error':'Business idea is required'}),400
    mode=d.get('mode','quick')
    if mode!='deep':
        data,err=gemini_analysis(d)
        if data is None:
            data=fallback(d)
            data['_provider']='Fallback'
            data['_notice']=err
        else: data['_provider']='Gemini'
        return jsonify(data)

    g,gerr=gemini_analysis(d); o,oerr=openai_analysis(d)
    if g is None and o is None:
        return jsonify({'error':'Deep Validation needs both GEMINI_API_KEY and OPENAI_API_KEY.','gemini_error':gerr,'openai_error':oerr}),502
    if g is None: return jsonify({'error':'Gemini analysis failed','detail':gerr}),502
    if o is None: return jsonify({'error':'OpenAI analysis failed','detail':oerr}),502
    synthesis,serr=synthesize(g,o)
    if synthesis is None:
        return jsonify({'error':'Final synthesis failed','detail':serr}),502
    return jsonify({'mode':'deep','gemini':g,'openai':o,'synthesis':synthesis,'_providers':['Gemini','OpenAI']})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')),debug=False)
