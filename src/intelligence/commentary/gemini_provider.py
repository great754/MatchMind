"""Gemini REST adapter, isolated from analytics and report code."""
import os
import re
from pathlib import Path
import requests
from dotenv import load_dotenv
from .base import CommentaryError

class GeminiCommentaryProvider:
    def __init__(self,model: str='gemini-3.5-flash-lite',session=None):
        load_dotenv(Path(__file__).resolve().parents[3] / '.env', override=False)
        self._key=os.environ.get('GEMINI_API_KEY','').strip()
        if not self._key: raise CommentaryError('GEMINI_API_KEY is missing. Set it in your environment or .env, or use --commentary-provider none.')
        if not re.fullmatch(r'[a-zA-Z0-9._-]+',model): raise CommentaryError('Invalid Gemini model identifier.')
        self.model=model
        self._session=session or requests.Session()
    @property
    def configuration(self):
        return {'provider':'gemini','model':self.model,'temperature':0.2,'max_output_tokens':512}
    def generate(self,request: dict) -> str:
        import json
        ids=list(request['sentences'])
        body={'systemInstruction':{'parts':[{'text':request['instruction']}]},
              'contents':[{'role':'user','parts':[{'text':json.dumps({k:v for k,v in request.items() if k!='instruction'},allow_nan=False)}]}],
              'generationConfig':{'temperature':0.2,'maxOutputTokens':512,'responseMimeType':'application/json',
                  'responseSchema':{'type':'OBJECT','properties':{'sentence_ids':{'type':'ARRAY','items':{'type':'STRING','enum':ids},'minItems':1,'maxItems':3}},'required':['sentence_ids']}}}
        try:
            response=self._session.post(f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
                headers={'x-goog-api-key':self._key,'Content-Type':'application/json'},json=body,timeout=(5,30),allow_redirects=False)
            if response.status_code!=200 or len(response.content)>1_000_000: raise ValueError()
            parts=response.json()['candidates'][0]['content']['parts']
            return ''.join(p['text'] for p in parts if 'text' in p and not p.get('thought'))
        except Exception:
            raise CommentaryError('Gemini request failed or returned an unusable response; check connectivity, model access and API quota.') from None
