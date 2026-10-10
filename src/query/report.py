"""Offline precomputed tool answers; browser receives no credentials or raw query access."""
import json
from . import MatchMindTools,query_question,explain_event
from .answers import answer_result
from .schema import QueryResult

SUGGESTIONS={
 'fastest':'Who was the fastest player?', 'possession':'Which team had more possession?',
 'difficult':'Show the most difficult completed transfer.', 'pressure':'When was pressure highest?',
 'network':'Which player connected with the most teammates?', 'formation':'What formation was detected?',
 'shape':'How did team shape change in possession?', 'block':'When did the teams use a high block?',
 'summary':'Give me a match summary.'}


def catalogue(output):
    tools=MatchMindTools(output)
    answers={}
    for kind,q in SUGGESTIONS.items():
        for team in ('all','left','right'):
            question=q if team=='all' else q+' '+team
            answers[kind+':'+team]=query_question(tools,question).model_dump(mode='json')
    answers['unsupported']=query_question(tools,'Was the player tired?').model_dump(mode='json')
    selected={}
    # One representative per event group avoids multiple explanations of the same transfer.
    for e in sorted(tools.events,key=lambda e:(e.type.value!='PASS_COMPLETED',e.timestamp)):
        if e.source in ('transfer','trajectory','possession') and e.type.value not in ('POSSESSION_GAINED','POSSESSION_LOST'):
            selected.setdefault(e.group_id,e)
    events=sorted(selected.values(),key=lambda e:(e.timestamp,e.event_id))
    explanations={e.event_id:explain_event(tools,e.event_id).model_dump(mode='json') for e in events}
    facts={}
    for answer in list(answers.values())+list(explanations.values()):
        result=QueryResult(question_type=answer['question_type'],status=answer['status'],facts=answer['facts'],limitations=answer['limitations'])
        answer['mode_text']={mode:[s.text for s in answer_result(result,mode).statements] for mode in ('beginner','fan','analyst')}
        for fact in answer.pop('facts',[]):facts[fact['fact_id']]=fact
    payload=dict(facts=facts,answers=answers,explanations=explanations,
        events=[dict(event_id=e.event_id,type=e.type.value,timestamp=e.timestamp,significance=e.significance) for e in events],
        suggestions=SUGGESTIONS,mode_notes={
            'beginner':'Possession means inferred control of the ball. A transfer is observed movement between players; it may not be an intentional pass.',
            'fan':'Transfers and tactical labels describe observed play; difficulty is a retrospective heuristic.',
            'analyst':'Source fields and fact IDs support the answer. BAS checks are not independent; earlier tactical evidence does not establish causality.'})
    # Also usable by a future provider-independent service.
    (output/'ask_matchmind.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n')
    return payload


def section():
    return '''<section aria-labelledby="ask-heading"><h2 id="ask-heading">Ask MatchMind</h2>
<p>Offline answers from deterministic MatchMind tools. Team 1 means left; Team 2 means right. Names, scores and tactical intentions are unavailable.</p>
<form id="ask-form"><label for="ask-question">Question</label> <input id="ask-question" type="text" maxlength="500" size="55" placeholder="Who was the fastest player?"> <button type="submit">Ask</button>
<label>Answer mode <select id="ask-mode"><option value="beginner">Beginner</option><option value="fan" selected>Fan</option><option value="analyst">Analyst</option></select></label></form>
<div id="ask-suggestions"></div><div id="ask-answer" role="status" aria-live="polite"></div>
<details><summary>Event explanation cards</summary><p>Explain a transfer, ownership change or shot candidate. Times use the current recording clock.</p><div id="ask-events" style="max-height:260px;overflow:auto"></div></details></section>'''


def script():
    return '''
const askData=data.askMatchMind,askInput=document.getElementById('ask-question'),askMode=document.getElementById('ask-mode'),askAnswer=document.getElementById('ask-answer');
let activeAsk=null;
function askIntent(q){if(/\\b(tired|fatigue|coach|intent|score|jersey|name|win|predict|weather)\\b/.test(q))return 'unsupported';if(/\\b(fastest|quickest)\\b/.test(q))return 'fastest';if(/\\b(hardest|difficult|difficulty)\\b/.test(q))return 'difficult';if(q.includes('possession'))return 'possession';if(q.includes('formation'))return 'formation';if(/\\b(pressure|pressured)\\b/.test(q))return 'pressure';if(/\\b(connect|connected|teammates|network)\\b/.test(q))return 'network';if(/\\b(shape|width|compactness)\\b/.test(q))return 'shape';if(q.includes('high block'))return 'block';if(/\\b(before|explain|turnover|shot|event)\\b/.test(q))return 'event';if(/\\b(summary|overview)\\b/.test(q))return 'summary';return 'unsupported';}
function showAsk(answer){activeAsk=answer;askAnswer.replaceChildren();const note=document.createElement('p');note.textContent=askData.mode_notes[askMode.value];askAnswer.appendChild(note);for(const [index,statement] of answer.statements.entries()){const p=document.createElement('p');p.textContent=answer.mode_text?.[askMode.value]?.[index]??statement.text;if(statement.timestamp!==null){const b=document.createElement('button');b.textContent=' '+statement.timestamp.toFixed(2)+'s';b.dataset.time=String(statement.timestamp);b.addEventListener('click',()=>{seek(statement.timestamp);highlightAsk(statement.event_id);});p.appendChild(b);}if(askMode.value==='analyst'&&statement.fact_ids.length){const evidence=document.createElement('small');evidence.textContent=' Facts: '+statement.fact_ids.map(id=>id+' ['+(askData.facts[id]?.source??'unavailable')+']').join(', ');p.appendChild(evidence);}askAnswer.appendChild(p);}const limits=document.createElement('details'),title=document.createElement('summary'),text=document.createElement('p');title.textContent='Evidence limits';text.textContent=answer.limitations.join(' ');limits.appendChild(title);limits.appendChild(text);askAnswer.appendChild(limits);}
function showSequence(event){const original=askData.explanations[event.event_id];const startFact=original.statements.flatMap(s=>s.fact_ids).map(id=>askData.facts[id]).find(f=>f.field==='start_frame');const start=startFact?startFact.value/data.fps:event.timestamp;const prior=askData.events.filter(e=>e.timestamp>=Math.max(0,start-15)&&e.timestamp<start).slice(-10);const extra=prior.map(e=>askData.explanations[e.event_id].statements[0]);const answer={...original,statements:[...extra,...original.statements],supporting_fact_ids:[...extra.flatMap(s=>s.fact_ids),...original.supporting_fact_ids],mode_text:{},limitations:[...original.limitations,'Recent events strictly precede release; this temporal sequence does not establish causality.']};for(const mode of ['beginner','fan','analyst'])answer.mode_text[mode]=[...prior.map(e=>askData.explanations[e.event_id].mode_text[mode][0]),...original.mode_text[mode]];showAsk(answer);}
function highlightAsk(eventId){for(const b of document.getElementById('ask-events').children){b.style.outline=b.dataset.eventId===eventId?'2px solid #1558b0':'';}}
function askQuestion(question){const q=question.toLowerCase().trim();let kind=askIntent(q);if(/\\b(red|blue)\\b/.test(q))kind='unsupported';const team=/\\b(team\\s*1|left)\\b/.test(q)?'left':/\\b(team\\s*2|right)\\b/.test(q)?'right':'all';if(kind==='event'){const match=q.match(/\\b(\\d+):(\\d+(?:\\.\\d+)?)\\b/);const t=match?Number(match[1])*60+Number(match[2]):Number(slider.value);const events=askData.events.filter(e=>q.includes('turnover')?e.type==='TRANSFER_INTERCEPTED':q.includes('shot')?e.type==='SHOT_CANDIDATE':true);let closest=null;for(const e of events){if(closest===null||Math.abs(e.timestamp-t)<Math.abs(closest.timestamp-t))closest=e;}if(closest&&(!match||Math.abs(closest.timestamp-t)<=2)){if(q.includes('before')||q.includes('sequence'))showSequence(closest);else showAsk(askData.explanations[closest.event_id]);highlightAsk(closest.event_id);return;}showAsk(askData.answers.unsupported);return;}showAsk(askData.answers[kind+':'+team]??askData.answers.unsupported);}
document.getElementById('ask-form').addEventListener('submit',event=>{event.preventDefault();askQuestion(askInput.value);});askMode.addEventListener('change',()=>{if(activeAsk)showAsk(activeAsk);});
for(const question of Object.values(askData.suggestions)){const b=document.createElement('button');b.textContent=question;b.style.margin='4px';b.addEventListener('click',()=>{askInput.value=question;askQuestion(question);});document.getElementById('ask-suggestions').appendChild(b);}
for(const event of askData.events){const b=document.createElement('button');b.dataset.eventId=event.event_id;b.style.display='block';b.style.margin='5px';b.textContent='Explain '+event.type.replaceAll('PASS','TRANSFER')+' · '+event.timestamp.toFixed(2)+'s';b.addEventListener('click',()=>{showAsk(askData.explanations[event.event_id]);highlightAsk(event.event_id);});document.getElementById('ask-events').appendChild(b);}
'''
