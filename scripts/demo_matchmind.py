"""Network-free eight-turn demo. Pass --provider for optional grounded selection."""
import argparse
import json
from src.artifacts import LocalArtifactStore
from src.query import MatchMindTools
from src.providers.agent import ask
from src.providers.factory import build_provider

QUESTIONS=['Who was the fastest player?','Tell me more about that player','Show me the most difficult completed transfer.','Why was it difficult?','Take me there.','When was Team 1 under the most pressure?','What formation did MatchMind detect?','What did the coach tell Team 1?']

def run(tools,recording,provider=None,name='local'):
    state={};answers=[]
    for question in QUESTIONS:
        response=ask(tools,question,provider=provider,conversation=state,recording_id=recording,requested_provider=name)
        state=response.conversation.model_dump(mode='json')
        answers.append(dict(question=question,response=response.model_dump(mode='json')))
    return answers

def main():
    from dotenv import load_dotenv
    load_dotenv(override=False)
    p=argparse.ArgumentParser();p.add_argument('--registry',default='configs/serving/local.json');p.add_argument('--recording',default='118575_clip');p.add_argument('--provider',choices=['local','gemini','microsoft','microsoft-framework'],default='local');a=p.parse_args()
    store=LocalArtifactStore(json.load(open(a.registry)));tools=MatchMindTools(store.resolve(a.recording))
    try:provider=build_provider(a.provider,tools)
    except Exception:provider=None
    print(json.dumps(run(tools,a.recording,provider,a.provider),indent=2))
if __name__=='__main__':main()
