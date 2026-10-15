"""Refresh only report UI source; preserve embedded measurements and commentary."""
import argparse
import json
from pathlib import Path
from src.query.report import section,script

def refresh(path):
    page=Path(path).read_text()
    start=page.index('<section aria-labelledby="ask-heading">');end=page.index('</section>',start)+len('</section>')
    page=page[:start]+section()+page[end:]
    start=page.index('const askData=data.askMatchMind');end=page.index("slider.addEventListener('input'",start)
    page=page[:start]+script()+page[end:]
    needle='if(data.owner[i]===data.ids[j]){'
    if "highlightedAskPlayer===data.ids[j]" not in page:
        page=page.replace(needle,"if(typeof highlightedAskPlayer!=='undefined'&&highlightedAskPlayer===data.ids[j]){ctx.strokeStyle='#00ffff';ctx.lineWidth=3;ctx.strokeRect(X(p[0])-11,Y(p[1])-11,22,22);}"+needle)
    Path(path).write_text(page)

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',default='configs/serving/local.json');a=p.parse_args()
    for directory in json.loads(Path(a.registry).read_text()).values():refresh(Path(directory)/'report.html')
if __name__=='__main__':main()
