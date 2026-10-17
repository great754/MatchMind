import {ArrowUpRight,ChevronRight,FileText} from 'lucide-react';
import type {Answer,Agent,Mode} from './contracts';
import {preciseTime} from './time';
import {useIdentity} from './identity';
import {factSentence} from './language';
export function Evidence({answer,agent,mode,onJump,title='Answer'}:{answer:Answer,agent?:Agent,mode:Mode,onJump:(time:number,event:string|null,actor:number|null)=>void,title?:string}){
 const names=useIdentity();
 const actor=answer.facts.find(f=>f.field==='player_id'&&f.source.startsWith('player_summary.csv'))??answer.facts.find(f=>f.field==='actor_id');
 const first=answer.statements.find(s=>s.timestamp!==null);const groups=['observed','inferred','validated','heuristic','unknown'];
 const preview=mode==='beginner'?2:mode==='fan'?3:answer.statements.length;
 const renderStatement=(s:Answer['statements'][number],i:number)=><p key={i}>{names.text(s.text)}{s.timestamp!==null&&<button className="inline-time" onClick={()=>onJump(s.timestamp!,s.event_id,typeof actor?.value==='number'?actor.value:null)}>{preciseTime(s.timestamp)} <ArrowUpRight size={12}/></button>}</p>;
 const facts=answer.facts.map(f=>({f,text:factSentence(f,names.player,names.team)})).filter(x=>x.text);
 return <div className="answer-block" aria-live="polite"><div className="answer-top"><span className="eyebrow">{title}</span><span className={`response-badge ${agent?.fallback?'fallback':''}`}>{agent?.fallback?'Local fallback':agent?.provider==='microsoft_foundry'?'Microsoft grounded selection':agent?.provider==='gemini'?'Gemini selection':'Grounded match evidence'}</span></div>
 {answer.status!=='ok'&&<p className="unknown-note">I don’t have enough evidence from this match to answer that reliably.</p>}
 {answer.status==='ok'&&<div className="answer-text">{answer.statements.slice(0,preview).map(renderStatement)}{answer.statements.length>preview&&<details className="limitations"><summary>More context</summary>{answer.statements.slice(preview).map(renderStatement)}</details>}</div>}

 {first&&<button className="primary jump" onClick={()=>onJump(first.timestamp!,first.event_id,typeof actor?.value==='number'?actor.value:null)}>Jump to moment <ArrowUpRight size={15}/></button>}
 {title==='Event explanation'&&<div className="explanation-kinds">{groups.map(group=>{const rows=agent?.evidence.filter(e=>e.category===group).map(e=>factSentence({...e,record_id:''},names.player,names.team)).filter(Boolean)??[];return <details key={group}><summary>{group==='validated'?'Annotation support':group==='observed'?'What was observed':group==='inferred'?'Tracking estimates':group==='heuristic'?'Tactical estimates':'What remains unknown'}</summary><p>{rows.length?rows.join(' '):'No separately supported conclusion is available in this category.'}</p></details>})}</div>}
 <details className="evidence-disclosure" open={mode==='analyst'}><summary><FileText size={14}/> Supporting evidence <ChevronRight size={13}/></summary><div className="fact-list">{facts.map(({f,text})=><div className="fact" key={f.fact_id}><p>{text}</p><small>{f.source.startsWith('player_summary')?'Source: player tracking summary':f.source.startsWith('transfers')?'Source: confirmed transfer records':f.source.startsWith('possession')?'Source: ball-control tracking':f.source.startsWith('formation')?'Source: four-minute formation review':'Source: match tracking and annotations'}</small></div>)}</div></details>
 <details className="limitations"><summary>Limitations & uncertainty</summary><p>These are estimates from recorded tracking. Missing observations can limit reliability, and the evidence does not establish coaching intent or causality.</p></details>
 </div>;
}
