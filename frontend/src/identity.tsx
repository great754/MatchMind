import {createContext,useContext,useEffect,useRef} from 'react';
import {z} from 'zod';
import type {Replay} from './contracts';
const side=z.enum(['left','right']);
const team=z.object({name:z.string().trim().min(1).max(60),color:z.string().regex(/^#[0-9a-fA-F]{6}$/)});
export const identitySchema=z.object({teams:z.object({left:team,right:team}),players:z.record(z.string(),z.object({name:z.string().trim().max(80),jersey:z.string().regex(/^\d{0,3}$/),team:side}))});
export type Identity=z.infer<typeof identitySchema>;
export const identityKey=(recordingId:string)=>`tactiqai:${recordingId}:identity:v1`;
export const IDENTITY_KEY=identityKey('118575_clip');
export const defaults:Identity={teams:{left:{name:'Home side',color:'#ff837b'},right:{name:'Away side',color:'#79a8ff'}},players:{}};
export function loadIdentity(recordingId='118575_clip'):Identity {try{const value=identitySchema.safeParse(JSON.parse(localStorage.getItem(identityKey(recordingId))??'null'));return value.success?value.data:structuredClone(defaults)}catch{return structuredClone(defaults)}}
export function saveIdentity(value:Identity,recordingId='118575_clip'){const parsed=identitySchema.parse(value);try{localStorage.setItem(identityKey(recordingId),JSON.stringify(parsed));return true}catch{return false}}
export const IdentityContext=createContext<{identity:Identity,ids:number[]}>({identity:defaults,ids:[]});
export function useIdentity(override?:{identity:Identity,ids:number[]}){const context=useContext(IdentityContext);const {identity,ids}=override??context;
 const team=(side:string)=>side in identity.teams?identity.teams[side as 'left'|'right'].name:'Unknown team';
 const player=(id:number)=>{const p=identity.players[String(id)];return (p?.name||(ids.includes(id)?`Player ${ids.indexOf(id)+1}`:'Unidentified player'))+(p?.jersey?` #${p.jersey}`:'')};
 const color=(side:string)=>side in identity.teams?identity.teams[side as 'left'|'right'].color:'#8793a7';
 const text=(value:string)=>value.replace(/Actor\s+(\d+)|\b(?:actor|player) id:\s*(\d+)/gi,(_,a,b)=>player(Number(a??b))).replace(/\b(?:Home side|Team 1|team: left)\b/g,()=>team('left')).replace(/\b(?:Away side|Team 2|team: right)\b/g,()=>team('right'));
 const canonicalQuestion=(question:string)=>{let q=question;for(const side of ['left','right'] as const){const name=identity.teams[side].name;q=q.replaceAll(name,side==='left'?'Team 1':'Team 2')}for(const [id,p] of Object.entries(identity.players)){if(p.name)q=q.replaceAll(p.name,`player ${id}`)}return q};
 return {identity,team,player,color,text,canonicalQuestion};
}
export function MatchSetup({value,replay,onSave,onClose}:{value:Identity,replay:Replay|null,onSave:(v:Identity)=>void,onClose:()=>void}){
 const panel=useRef<HTMLElement>(null);
 useEffect(()=>{const previous=document.activeElement as HTMLElement|null;panel.current?.querySelector<HTMLInputElement>('input')?.focus();return()=>previous?.focus()},[]);
 return <div className="setup-backdrop" onKeyDown={e=>{if(e.key==='Escape'){e.preventDefault();onClose()}if(e.key==='Tab'){const elements=Array.from(panel.current?.querySelectorAll<HTMLElement>('button,input,select,summary')??[]).filter(el=>el.getClientRects().length);const first=elements[0],last=elements.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}}}}><section ref={panel} className="setup-panel" role="dialog" aria-modal="true" aria-labelledby="setup-title"><h2 id="setup-title">Set up your match</h2><p>Give the teams and tracked players familiar names. These labels are saved only in this browser; tracking and analytics stay unchanged.</p><form onSubmit={e=>{e.preventDefault();const form=new FormData(e.currentTarget);const teams={...value.teams};for(const side of ['left','right'] as const)teams[side]={name:String(form.get(side+'-name')),color:String(form.get(side+'-color'))};const players={...value.players};for(const id of replay?.ids??[])players[id]={name:String(form.get(id+'-name')??''),jersey:String(form.get(id+'-jersey')??''),team:form.get(id+'-team') as 'left'|'right'};onSave(identitySchema.parse({teams,players}));}}>
 <div className="setup-teams">{(['left','right'] as const).map((side,i)=><fieldset key={side}><legend>{i===0?'Home':'Away'} team</legend><label>Team name<input name={side+'-name'} defaultValue={value.teams[side].name} required maxLength={60}/></label><label>Team color<input name={side+'-color'} type="color" defaultValue={value.teams[side].color}/></label></fieldset>)}</div>
 <details><summary>Name your players</summary><p>Use the tracking reference only to match a player. Blank names receive neutral player labels.</p><div className="setup-roster">{replay?.ids.map((id,i)=><fieldset key={id}><legend>Player {i+1} <small>Tracking reference {id}</small></legend><label>Name<input name={id+'-name'} defaultValue={value.players[id]?.name??''} maxLength={80}/></label><label>Jersey<input name={id+'-jersey'} defaultValue={value.players[id]?.jersey??''} inputMode="numeric" pattern="[0-9]{0,3}" maxLength={3}/></label><label>Team<select name={id+'-team'} defaultValue={value.players[id]?.team??replay.teams[i]}><option value="left">{value.teams.left.name}</option><option value="right">{value.teams.right.name}</option></select></label></fieldset>)}</div></details>
 <div className="setup-actions"><button type="button" className="secondary" onClick={onClose}>Use neutral labels</button><button className="primary">Save match setup</button></div></form></section></div>;
}
