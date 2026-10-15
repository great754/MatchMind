"""Run post-hoc diagnostics over an existing frozen evaluation directory."""
import argparse
import json
from pathlib import Path
from .artifacts import load_evaluation_recording
from .shot_diagnostics import analyze_shot_errors,write_shot_diagnostics,report_section
from .__main__ import write_report
from .transfer_diagnostics import analyze_transfer_errors,write_transfer_diagnostics
import html


def run(output,data_root=Path('data/soccertrack')):
    output=Path(output);payload=json.loads((output/'evaluation_summary.json').read_text());rows=[];transfer_rows=[];classes=[]
    for item in payload['recordings']:
        print('Diagnosing',item['recording_id'],flush=True)
        recording,result,config=load_evaluation_recording(output/'per_match'/item['recording_id'],data_root)
        rows.extend(analyze_shot_errors(recording,result,config,item['recording_id']))
        errors,class_result=analyze_transfer_errors(recording,result,config,item['recording_id'])
        transfer_rows.extend(errors);classes.append(class_result)
    diagnostics=write_shot_diagnostics(rows,output)
    transfers=write_transfer_diagnostics(transfer_rows,classes,output)
    write_report(payload,output)
    path=output/'evaluation_report.html';page=path.read_text();path.write_text(page.replace('</html>','<p><a href="shot_experiments.html">Controlled shot experiments</a> · <a href="shot_error_analysis.json">Shot diagnostic JSON</a> · <a href="shot_error_analysis.csv">Shot diagnostic CSV</a></p>'+report_section(diagnostics)+'<h2>Transfer error analysis</h2><p>No transfer thresholds were changed. Full row evidence: <a href="transfer_error_analysis.json">JSON</a> · <a href="transfer_error_analysis.csv">CSV</a></p><pre>'+html.escape(json.dumps({k:v for k,v in transfers.items() if k!="items"},indent=2))+'</pre></html>'))
    return diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    parser.add_argument('--data-root',type=Path,default=Path('data/soccertrack'))
    args=parser.parse_args();result=run(args.output,args.data_root)
    print(json.dumps({k:result[k] for k in ('counts','false_negative_gate_counts','false_positive_hypotheses')},indent=2))

if __name__=='__main__':main()
