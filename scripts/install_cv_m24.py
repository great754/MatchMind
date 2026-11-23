"""Install an optional read-only benchmark only after validating pinned evidence."""
import argparse,json,tempfile
from pathlib import Path
from src.serving.cv import digest
from src.serving.footandball24 import Benchmark
from src.vision.intelligent23.inputs import write


def install(inputs,evaluation,config):
    data={'report':str(Path(evaluation)/'report.json'),'report_sha256':digest(Path(evaluation)/'report.json'),'inputs':str(Path(inputs)/'manifest.json')}
    with tempfile.TemporaryDirectory() as directory:
        temporary=Path(directory)/'config.json';write(temporary,data);benchmark=Benchmark(temporary)
        if set(benchmark.files)!={'giroud','test_match','team_goal','southampton'}:raise ValueError('M24 requires exactly four approved short clips')
        for name in benchmark.files:benchmark.frame(name,benchmark.inputs['sources'][name]['frames']-1)
    write(config,data);print(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--config',default='configs/serving/cv_m24.json');a=p.parse_args();install(a.inputs,a.evaluation,a.config)
