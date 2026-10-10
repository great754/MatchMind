import argparse
import json
from . import MatchMindTools,query_question

def main():
    parser=argparse.ArgumentParser(description='Ask deterministic MatchMind tools about an exported recording.')
    parser.add_argument('--recording',required=True)
    parser.add_argument('--question',required=True)
    parser.add_argument('--mode',choices=('beginner','fan','analyst'),default='fan')
    parser.add_argument('--timestamp',type=float,default=0)
    args=parser.parse_args()
    print(json.dumps(query_question(MatchMindTools(args.recording),args.question,args.mode,args.timestamp).model_dump(mode='json'),indent=2))

if __name__=='__main__':main()
