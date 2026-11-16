"""Reversible user annotations; immutable source tracking is never rewritten."""
import json,os,threading,time
from pathlib import Path
from fastapi import HTTPException
from src.highlights.index import digest

class ConfirmedIdentityStore:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()
    def path(self,recording,session):return self.root/(digest([recording,str(session)])+'.json')
    def load(self,recording,session,catalog):
        with self.lock:
            p=self.path(recording,session);data=json.loads(p.read_text()) if p.exists() else {'revision':0,'players':{}}
            data['players']={ref:v for ref,v in data['players'].items() if ref in catalog.by_ref and v['source_signature']==catalog.by_ref[ref]['source_signature']}
            return data
    def label(self,ref,data,catalog):
        p=data['players'].get(ref);fallback=catalog.by_ref.get(ref,{}).get('label','This player')
        if p:return (p['name'] or fallback)+(f" #{p['jersey']}" if p['jersey'] is not None else '')
        return fallback
    def confirm(self,recording,body,catalog,portraits=None):
        with self.lock:
            if body.player_ref not in catalog.by_ref:raise HTTPException(422,'Choose a player from this recording')
            target=catalog.by_ref[body.player_ref]
            if body.image_id and (not portraits or body.image_id not in [c['image_id'] for c in portraits.get(body.player_ref,[])]):raise HTTPException(422,'That image belongs to a different player')
            data=self.load(recording,body.session_id,catalog)
            for ref,v in data['players'].items():
                if ref!=body.player_ref and body.jersey is not None and v['jersey']==body.jersey and v['team']==body.team and max(v['start'],target['start'])<min(v['end'],target['end']):raise HTTPException(409,'That number is already assigned to another player on this team during the same time. Review the two cards.')
            data['revision']+=1;data['players'][body.player_ref]={'jersey':body.jersey,'name':body.name.strip(),'team':body.team,'role':body.role,'source_signature':target['source_signature'],'start':target['start'],'end':target['end'],'image_id':body.image_id,'provenance':'USER_CONFIRMED','confirmed_at':time.time()}
            p=self.path(recording,body.session_id);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,p);return data
    def save_teams(self,recording,body,catalog):
        if body.left.name.strip().casefold()==body.right.name.strip().casefold():raise HTTPException(409,'Use different names for the two teams.')
        with self.lock:
            data=self.load(recording,body.session_id,catalog);data['teams']={'left':body.left.model_dump(),'right':body.right.model_dump()};data['revision']+=1;p=self.path(recording,body.session_id);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data));os.replace(tmp,p);return data
    def remove(self,recording,session,ref,catalog):
        with self.lock:
            data=self.load(recording,session,catalog);data['players'].pop(ref,None);data['revision']+=1;p=self.path(recording,session);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data));os.replace(tmp,p);return data
    def resolve(self,text,data,catalog,selected=None):
        import re
        refs=[]
        for ref,player in catalog.by_ref.items():
            if re.search(r'(?<!\w)'+re.escape(player['label'])+r'(?!\w)',text,re.I):refs.append(ref)
        for ref,p in data['players'].items():
            if p['name'] and re.search(r'(?<!\w)'+re.escape(p['name'])+r'(?!\w)',text,re.I):refs.append(ref)
        numbers=re.findall(r'\b(?:player|number|jersey)\s*#?(\d+)\b|#(\d+)\b',text,re.I)
        comparison=re.search(r'\bcompare player\s*(\d+)\s+(?:and|with)\s*(?:player\s*)?(\d+)',text,re.I)
        if comparison:numbers=[(comparison[1],''),(comparison[2],'')]
        for a,b in numbers:
            matches=[ref for ref,p in data['players'].items() if p['jersey']==int(a or b)]
            if len(matches)!=1:return [],'Please confirm that player’s number in Match setup first. If both teams use that number, select the player’s card.'
            refs+=matches
        if not refs and selected and re.search(r'\b(this player|selected player|they|their|his|her|he|she|stats|statistics|player profile)\b',text,re.I):
            if selected not in catalog.by_ref:return [],'Please select a player in this recording first.'
            refs=[selected]
        return list(dict.fromkeys(refs)),None
