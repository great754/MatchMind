"""Export new demo sidecars from existing canonical artifacts, without detectors."""
import argparse
import json
from pathlib import Path
from src.query import MatchMindTools
from src.tactics.review import pressure_level,review_pressure,formation_windows
from src.serving.review import load_replay
from src.analytics.possession import attribute_possession


def prepare(directory):
    directory=Path(directory);tools=MatchMindTools(directory)
    ownership=attribute_possession(tools._csv('possession.csv'),tools.summary.get('fps',25))
    replay,_=load_replay(directory);pressure=review_pressure(replay,ownership)
    values={'possession_attribution.json':ownership,'formation_windows.json':formation_windows(tools.states),
            'pressure_levels.json':[dict(timestamp=i*replay.stride/replay.fps,level=pressure_level(p),**p.model_dump()) for i,p in enumerate(pressure)]}
    for name,value in values.items():(directory/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return list(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print('Prepared: '+', '.join(prepare(args.directory)))

if __name__=='__main__':main()
