"""Human-readable text from deterministic evidence; raw details remain inspectable."""
import re
class ResponseComposer:
    @staticmethod
    def readable(text,catalog,identities,store):
        def actor(m):
            ref='actor:'+m.group(1);return store.label(ref,identities,catalog) if ref in catalog.by_ref else 'an unidentified player'
        text=re.sub(r'\bActor\s+(\d+)\b',actor,text,flags=re.I)
        text=re.sub(r'\b(?:actor|player) id:\s*(\d+)\b',actor,text,flags=re.I)
        terms={'observable transfers':'recorded passes','observable transfer':'recorded pass','canonical evidence':'match evidence','Home side':'Team A','Away side':'Team B','BAS':'event annotations','MOT':'video observations','GSR':'pitch positions','heuristic':'estimated','retrospective':'recorded','inverse geometric difficulty':'estimated difficulty','sustained closing-pressure':'pressing'}
        for old,new in terms.items():text=re.sub(r'\b'+re.escape(old)+r'\b',new,text,flags=re.I)
        text=re.sub(r'\b(?:evt_[a-zA-Z0-9_]+|TRACK_[a-zA-Z0-9_:]+|PLAYER_\d+|actor:\d+)\b','',text)
        return re.sub(r'\s+',' ',text).strip()

    @staticmethod
    def player_summary(name,metrics):
        values={m['name']:m['value'] for m in metrics}
        shots=values.get('Recorded shots');goals=values.get('Recorded goals');passes=values.get('Completed pass estimates');attempts=values.get('Recorded pass attempts');accuracy=values.get('Estimated pass completion');tackles=values.get('Recorded tackles');interceptions=values.get('Interception estimates')
        parts=[]
        if shots is not None:parts.append(f"took {int(shots)} shot{'s' if shots!=1 else ''}")
        if goals is not None:parts.append(f"scored {int(goals)} goal{'s' if goals!=1 else ''}")
        if passes is not None:
            passing=f"completed {int(passes)}"+(f" of {int(attempts)} passes" if attempts is not None else ' passes')
            if accuracy is not None:passing+=f" ({accuracy:.1f}% estimated accuracy)"
            parts.append(passing)
        text=name+' '+', '.join(parts)+'.' if parts else name+' has no supported statistics in this footage.'
        defensive=[]
        if tackles is not None:defensive.append(f"{int(tackles)} tackle{'s' if tackles!=1 else ''}")
        if interceptions is not None:defensive.append(f"{int(interceptions)} estimated interception{'s' if interceptions!=1 else ''}")
        if defensive:text+=' The records also show '+' and '.join(defensive)+'.'
        return text+' These figures cover the available footage; passes and interceptions are tracking estimates.'
