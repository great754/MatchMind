"""Standalone plots and an offline review page; no web service is required."""
import html
import json
import os
from pathlib import Path

import numpy as np


def _number(value):
    if value is None:
        return None
    if isinstance(value,np.generic):
        value = value.item()
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def _nested(array):
    a = np.asarray(array,dtype=float)
    return np.where(np.isfinite(a),np.round(a,3),None).tolist()


def _table(frame, columns, max_rows=100):
    rows = []
    for _,r in frame.head(max_rows).iterrows():
        cells = []
        for column in columns:
            value = r[column]
            value = _number(value)
            if value is None or str(value) == '<NA>':
                text = '—'
            elif isinstance(value,float):
                text = f'{value:.2f}'
            else:
                text = str(value)
            if column in ('start_seconds','seconds') and value is not None:
                text = f'<button data-seek="{float(value):.3f}">{html.escape(text)}s</button>'
            else:
                text = html.escape(text)
            cells.append(f'<td>{text}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    headers = ''.join(f'<th>{html.escape(c.replace("_"," "))}</th>' for c in columns)
    return '<div class="scroll"><table><thead><tr>'+headers+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def create_report(recording,result,summaries,summary,output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matchmind-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time = np.arange(len(recording.ball))/recording.fps
    fig,axes = plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
    ball = result['ball']
    axes[0].plot(time,ball.instantaneous_speed_mps,color='#aaaaaa',lw=.5,label='Raw (quantized)')
    axes[0].plot(time,ball.smoothed_speed_mps,color='#2369c9',lw=.9,label='Smoothed, accepted')
    axes[0].set(ylabel='Ball speed (m/s)',ylim=(0, min(80,max(20,summary['config']['ball_speed_limit_mps']*1.2))))
    axes[0].legend(loc='upper right')
    for e in recording.events:
        if e['label'] == 'SHOT':
            axes[0].axvline(e['seconds'],color='#dd3333',alpha=.7,ls='--',lw=.8)
    for _,row in summaries.nlargest(3,'measured_distance_m').iterrows():
        slot = list(recording.player_ids).index(int(row.player_id))
        m = result['players'][slot]
        axes[1].plot(time,m.smoothed_speed_mps,lw=.7,label=str(int(row.player_id)))
    axes[1].set(ylabel='Player speed (m/s)')
    axes[1].legend(title='Most measured distance',loc='upper right',ncol=3)
    owner = result['possession'].possessing_team.map({'left':1,'right':2}).fillna(0).to_numpy()
    axes[2].step(time,owner,where='post',color='#667077',lw=.7)
    axes[2].set(ylabel='Controlled team',xlabel='Recording time (s)',yticks=[0,1,2],yticklabels=['none','left','right'])
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('Estimated 2D motion and inferred possession; red dashed lines are BAS shots')
    fig.savefig(output/'motion.png',dpi=140)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(12,8),constrained_layout=True)
    ax.set_facecolor('#edf5ee')
    ax.plot([0,105,105,0,0],[0,0,68,68,0],color='#335d3d')
    ax.plot([52.5,52.5],[0,68],color='#335d3d',lw=.8)
    ax.add_patch(plt.Circle((52.5,34),9.15,fill=False,color='#335d3d',lw=.8))
    colors = {'completed':'#286eb9','intercepted':'#dd5339','out_of_play':'#946c32','unresolved':'#999999'}
    for r in result['passes'].itertuples(index=False):
        ax.annotate('',(r.end_x_m,r.end_y_m),(r.start_x_m,r.start_y_m),
                    arrowprops=dict(arrowstyle='->',color=colors[r.outcome],alpha=.55,lw=1))
    for label,color in colors.items():
        ax.plot([],[],color=color,label=label)
    ax.set(xlim=(-5,110),ylim=(73,-5),xlabel='Pitch x (m)',ylabel='Pitch y (m)',
           title='Observable ball-transfer candidates (not all are necessarily intended passes)')
    ax.set_aspect('equal')
    ax.legend()
    fig.savefig(output/'passes.png',dpi=140)
    plt.close(fig)

    # Small 5 Hz review snapshots; the exported metrics still use every 25 Hz frame.
    indices = np.arange(0,len(time),5)
    rows = result['possession'].iloc[indices]
    payload = dict(fps=recording.fps,stride=5,frames=len(time),ids=recording.player_ids.astype(int).tolist(),
                   teams=recording.player_teams,sideColors=recording.sync['side_to_color'],
                   positions=_nested(result['positions'][indices]),ball=_nested(ball.positions[indices]),
                   speed=_nested(np.stack([m.smoothed_speed_mps[indices] for m in result['players']],axis=1)),
                   ballSpeed=_nested(ball.smoothed_speed_mps[indices]),
                   owner=[None if str(x)=='<NA>' else int(x) for x in rows.possessing_player_id],
                   states=rows.state.tolist())
    video = ''
    source_video = Path('data/soccertrack/mot/clips')/f"{recording.sync['match_id']}.mp4"
    if recording.scope == 'clip' and source_video.exists():
        relative = os.path.relpath(source_video.resolve(),output.resolve())
        video = f'<video id="video" controls preload="metadata" src="{html.escape(relative)}"></video>'
    max_time = max(0,(len(time)-1)/recording.fps)
    sections = [f'<h2>Players</h2>{_table(summaries,["player_id","mot_id","team","measured_distance_m","max_speed_kmh","valid_motion_fraction","controlled_seconds"])}',
                f'<h2>Transfers</h2><p>First 100 candidates; complete results are in passes.csv. Difficulty is a heuristic, not a completion probability.</p>{_table(result["passes"],["pass_id","start_seconds","sender_id","receiver_id","outcome","distance_m","duration_seconds","ball_speed_peak_kmh","difficulty_score_0_100"])}',
                f'<h2>Trajectory shot candidates</h2>{_table(result["shots"],["shot_id","start_seconds","shooter_id","team","speed_peak_kmh","speed_median_mps","window_coverage_fraction"])}',
                f'<h2>BAS reference shot windows</h2>{_table(result["bas_shots"],["seconds","player_id","team","speed_peak_kmh","speed_median_mps","window_coverage_fraction"])}']
    validation = html.escape(json.dumps({'passes':summary['pass_validation'],'shots':summary['shot_validation']},indent=2))
    limitations = ''.join(f'<li>{html.escape(s)}</li>' for s in summary['limitations'])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>MatchMind analytics review</title>
<style>body{max-width:1250px;margin:30px auto;padding:0 20px;background:#f8fafc;color:#233044;font:15px system-ui;line-height:1.55}h1,h2{color:#17283b}video,img{width:100%;border-radius:8px}canvas{width:100%;max-width:820px;background:#245236;border-radius:8px}input[type=range]{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #dce3ea;white-space:nowrap;text-align:left;font-size:13px}button{cursor:pointer;color:#1558b0;border:0;background:#eaf2ff;padding:5px 8px;border-radius:4px}pre{white-space:pre-wrap;background:white;padding:20px;font-size:13px}.note{background:#fff4db;padding:16px;border-radius:8px}a{color:#1558b0}</style>
<h1>MatchMind analytics review</h1>'''
    page += f'<p>Match {html.escape(str(recording.sync["match_id"]))} · {recording.scope} · {len(time)} frames · {recording.fps:g} FPS · {recording.player_source.upper()} coordinates</p>'
    page += '<p class="note">These are estimates from coarse positions and an event-interpolated ball path. BAS matching is a consistency check, not independent accuracy. Blank speeds mean unavailable evidence.</p>'
    page += '<h2>Review positions and possession</h2><p>Scrub the timeline or click an event time. Canvas snapshots update at 5 Hz; metrics are calculated at 25 Hz.</p>'+video
    page += f'<input id="time" type="range" min="0" max="{max_time}" step="0.04" value="0"><p id="state"></p><label><input id="labels" type="checkbox" checked> Player IDs and km/h</label><br><canvas id="pitch" width="1100" height="750"></canvas>'
    page += '<h2>Motion</h2><img src="motion.png" alt="Motion and ownership over time"><h2>Transfer map</h2><img src="passes.png" alt="Transfer candidates on pitch">'
    page += ''.join(sections)+'<h2>Validation</h2><pre>'+validation+'</pre><h2>Limitations</h2><ul>'+limitations+'</ul>'
    page += '<p><a href="summary.json">Configuration and provenance</a> · <a href="possession.csv">Per-frame possession</a> · <a href="ball_frames.csv">Ball metrics</a> · <a href="passes.csv">All transfers and difficulty features</a> · <a href="shots.csv">Shot candidates</a></p>'
    page += '<script>const data='+json.dumps(payload,allow_nan=False,separators=(',',':'))+';'
    page += '''const slider=document.getElementById('time'),video=document.getElementById('video'),canvas=document.getElementById('pitch'),ctx=canvas.getContext('2d');
const X=x=>25+x*10,Y=y=>35+y*10;
function draw(t){const i=Math.min(data.positions.length-1,Math.max(0,Math.floor(t*data.fps/data.stride)));ctx.clearRect(0,0,1100,750);ctx.strokeStyle='#cfdfcf';ctx.lineWidth=2;ctx.strokeRect(X(0),Y(0),1050,680);ctx.beginPath();ctx.moveTo(X(52.5),Y(0));ctx.lineTo(X(52.5),Y(68));ctx.stroke();ctx.beginPath();ctx.arc(X(52.5),Y(34),91.5,0,Math.PI*2);ctx.stroke();ctx.strokeRect(X(0),Y(13.84),165,403.2);ctx.strokeRect(X(88.5),Y(13.84),165,403.2);ctx.strokeRect(X(0),Y(24.84),55,183.2);ctx.strokeRect(X(99.5),Y(24.84),55,183.2);
for(let j=0;j<data.ids.length;j++){const p=data.positions[i][j];if(p[0]===null||p[1]===null)continue;let side=data.teams[j]==='left'?'0':'1';ctx.fillStyle=data.sideColors[side]===0?'#4695ff':'#ff6363';ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),7,0,Math.PI*2);ctx.fill();if(data.owner[i]===data.ids[j]){ctx.strokeStyle='#ffdf35';ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),12,0,Math.PI*2);ctx.stroke();}if(document.getElementById('labels').checked){ctx.fillStyle='#ffffff';ctx.font='12px system-ui';let s=data.speed[i][j];ctx.fillText(data.ids[j]+' / '+(s===null?'—':(s*3.6).toFixed(1)),X(p[0])+10,Y(p[1])-5);}}
let b=data.ball[i];if(b[0]!==null&&b[1]!==null){ctx.fillStyle='#fff';ctx.strokeStyle='#111';ctx.beginPath();ctx.arc(X(b[0]),Y(b[1]),6,0,Math.PI*2);ctx.fill();ctx.stroke();}let speed=data.ballSpeed[i];document.getElementById('state').textContent=t.toFixed(2)+'s | '+data.states[i]+' | owner '+(data.owner[i]??'none')+' | ball '+(speed===null?'unavailable':(speed*3.6).toFixed(1)+' km/h (estimated)');}
function seek(t){slider.value=t;if(video)video.currentTime=t;draw(t);}
slider.addEventListener('input',()=>seek(Number(slider.value)));document.getElementById('labels').addEventListener('change',()=>draw(Number(slider.value)));document.querySelectorAll('[data-seek]').forEach(b=>b.addEventListener('click',()=>seek(Number(b.dataset.seek))));if(video)video.addEventListener('timeupdate',()=>{slider.value=video.currentTime;draw(video.currentTime);});draw(0);</script></html>'''
    (output/'report.html').write_text(page)
