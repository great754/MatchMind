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
            if column=='outcome' and value=='intercepted':text='opposite-team receipt (interception candidate)'
            if column in ('start_seconds','seconds') and value is not None:
                text = f'<button data-seek="{float(value):.3f}">{html.escape(text)}s</button>'
            else:
                text = html.escape(text)
            cells.append(f'<td>{text}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    labels={'pass_id':'transfer candidate ID','ball_speed_peak_kmh':'peak estimated 2D ball speed (km/h)',
            'speed_peak_kmh':'peak estimated 2D ball speed in window (km/h)',
            'speed_median_mps':'median estimated 2D ball speed in window (m/s)',
            'difficulty_score_0_100':'retrospective heuristic difficulty (0–100)',
            'max_speed_kmh':'peak estimated player speed (km/h)'}
    headers = ''.join(f'<th>{html.escape(labels.get(c,c.replace("_"," ")))}</th>' for c in columns)
    return '<div class="scroll"><table><thead><tr>'+headers+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def create_report(recording,result,summaries,summary,output,intelligence=None):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matchmind-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time = np.arange(len(recording.ball))/recording.fps
    fig,axes = plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
    ball = result['ball']
    axes[0].plot(time,ball.instantaneous_speed_mps,color='#aaaaaa',lw=.5,label='Raw (quantized)')
    axes[0].plot(time,ball.smoothed_speed_mps,color='#2369c9',lw=.9,label='Smoothed, accepted')
    axes[0].set(ylabel='Estimated 2D ball speed (m/s)',ylim=(0, min(80,max(20,summary['config']['ball_speed_limit_mps']*1.2))))
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
        ax.plot([],[],color=color,label='opposite-team receipt' if label=='intercepted' else label)
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
    intelligence = intelligence or {'events':{'events':[],'player_labels':{}},'commentary':{'items':[],'provider':{'provider':'none'},'errors':[]}}
    payload['intelligence']=intelligence
    payload['displayLabels']=intelligence['events']['player_labels']
    from src.query.report import catalogue,section as ask_section,script as ask_script
    payload['askMatchMind']=catalogue(output)
    video = ''
    source_video = Path(getattr(recording,'video_path',Path('data/soccertrack/mot/clips')/f"{recording.sync['match_id']}.mp4"))
    if recording.scope == 'clip' and source_video.exists():
        relative = os.path.relpath(source_video.resolve(),output.resolve())
        video = f'<video id="video" controls preload="metadata" src="{html.escape(relative)}"></video>'
    max_time = max(0,(len(time)-1)/recording.fps)
    sections = [f'<h2>Players</h2>{_table(summaries,["player_id","mot_id","team","measured_distance_m","max_speed_kmh","valid_motion_fraction","controlled_seconds"])}',
                f'<h2>Transfers</h2><p>First 100 candidates; complete results and provenance are in transfers.csv (passes.csv retained for compatibility). Difficulty is a heuristic, not a completion probability.</p>{_table(result["passes"],["pass_id","start_seconds","sender_id","receiver_id","outcome","distance_m","duration_seconds","ball_speed_peak_kmh","difficulty_score_0_100"])}',
                f'<h2>Trajectory shot candidates</h2>{_table(result["shots"],["shot_id","start_seconds","shooter_id","team","speed_peak_kmh","speed_median_mps","window_coverage_fraction"])}',
                f'<h2>BAS reference shot windows: estimated 2D ball motion</h2>{_table(result["bas_shots"],["seconds","player_id","team","speed_peak_kmh","speed_median_mps","window_coverage_fraction"])}']
    validation = html.escape(json.dumps({'transfer_consistency':summary['pass_validation'],'shots':summary['shot_validation']},indent=2))
    limitations = ''.join(f'<li>{html.escape(s)}</li>' for s in summary['limitations'])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>MatchMind analytics review</title>
<style>body{max-width:1250px;margin:30px auto;padding:0 20px;background:#f8fafc;color:#233044;font:15px system-ui;line-height:1.55}h1,h2{color:#17283b}video,img{width:100%;border-radius:8px}canvas{width:100%;max-width:820px;background:#245236;border-radius:8px}input[type=range]{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #dce3ea;white-space:nowrap;text-align:left;font-size:13px}button{cursor:pointer;color:#1558b0;border:0;background:#eaf2ff;padding:5px 8px;border-radius:4px}pre{white-space:pre-wrap;background:white;padding:20px;font-size:13px}.note{background:#fff4db;padding:16px;border-radius:8px}a{color:#1558b0}</style>
<h1>MatchMind analytics review</h1>'''
    page += f'<p>Match {html.escape(str(recording.sync["match_id"]))} · {recording.scope} · {len(time)} frames · {recording.fps:g} FPS · {recording.player_source.upper()} coordinates</p>'
    page += '<p class="note">These are estimates from coarse positions and an event-interpolated ball path. BAS matching is a consistency check, not independent accuracy. Blank speeds mean unavailable evidence.</p>'
    page += '<h2>Review positions and possession</h2><p>Scrub the timeline or click an event time. Canvas snapshots update at 5 Hz; metrics are calculated at 25 Hz.</p>'+video
    page += f'<input id="time" type="range" min="0" max="{max_time}" step="0.04" value="0"><p id="state"></p><label><input id="labels" type="checkbox" checked> Player labels and km/h</label><br><canvas id="pitch" width="1100" height="750"></canvas><p>Possession: <strong id="possession-team" aria-live="polite">No confirmed possession</strong></p>'
    page += ask_section()
    page += '<h2>Match intelligence</h2><p id="commentary-status"></p><label>Commentary mode <select id="commentary-mode"><option value="beginner">Beginner</option><option value="fan">Fan</option><option value="analyst">Analyst</option></select></label> <label><input type="checkbox" id="callouts" checked> Speed and measured-distance callouts</label><p id="live-callouts"></p><div id="commentary-list"></div><p><a href="events.json">Canonical events</a> · <a href="rolling_context.json">Rolling context</a> · <a href="commentary.json">Commentary metadata</a></p>'
    from src.tactics.report import section,script as tactics_script
    page += section(intelligence.get('tactics'))
    page += '<h2>Motion</h2><img src="motion.png" alt="Motion and ownership over time"><h2>Transfer map</h2><img src="passes.png" alt="Transfer candidates on pitch">'
    page += ''.join(sections)+'<h2>Validation</h2><pre>'+validation+'</pre><h2>Limitations</h2><ul>'+limitations+'</ul>'
    page += '<p><a href="summary.json">Configuration and provenance</a> · <a href="possession.csv">Per-frame possession</a> · <a href="ball_frames.csv">Ball metrics</a> · <a href="transfers.csv">All transfers and heuristic difficulty features</a> · <a href="shots.csv">Shot candidates</a></p>'
    page += '<script>const data='+json.dumps(payload,allow_nan=False,separators=(',',':')).replace('<',r'\u003c').replace('>',r'\u003e').replace('&',r'\u0026')+';'
    page += '''const slider=document.getElementById('time'),video=document.getElementById('video'),canvas=document.getElementById('pitch'),ctx=canvas.getContext('2d');
const X=x=>25+x*10,Y=y=>35+y*10;
function updatePossession(i){
 const label=document.getElementById('possession-team');
 const slot=data.owner[i]===null?-1:data.ids.indexOf(data.owner[i]);
 const team=slot<0?null:data.teams[slot];
 const side=team==='left'?'0':team==='right'?'1':null;
 const color=side===null?null:data.sideColors[side];
 if(data.states[i]==='controlled'&&(color===0||color===1)){
  label.textContent=color===0?'● Blue team':'● Red team';
  label.style.color=color===0?'#4695ff':'#ff6363';
 }else{
  label.textContent=data.states[i]==='contested'?'Contested':data.states[i]==='out_of_play'?'Out of play':'No confirmed possession';
  label.style.color='#64748b';
 }
}
function draw(t){const i=Math.min(data.positions.length-1,Math.max(0,Math.floor(t*data.fps/data.stride)));ctx.clearRect(0,0,1100,750);ctx.strokeStyle='#cfdfcf';ctx.lineWidth=2;ctx.strokeRect(X(0),Y(0),1050,680);ctx.beginPath();ctx.moveTo(X(52.5),Y(0));ctx.lineTo(X(52.5),Y(68));ctx.stroke();ctx.beginPath();ctx.arc(X(52.5),Y(34),91.5,0,Math.PI*2);ctx.stroke();ctx.strokeRect(X(0),Y(13.84),165,403.2);ctx.strokeRect(X(88.5),Y(13.84),165,403.2);ctx.strokeRect(X(0),Y(24.84),55,183.2);ctx.strokeRect(X(99.5),Y(24.84),55,183.2);
for(let j=0;j<data.ids.length;j++){const p=data.positions[i][j];if(p[0]===null||p[1]===null)continue;let side=data.teams[j]==='left'?'0':'1';ctx.fillStyle=data.sideColors[side]===0?'#4695ff':'#ff6363';ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),7,0,Math.PI*2);ctx.fill();if(data.owner[i]===data.ids[j]){ctx.strokeStyle='#ffdf35';ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),12,0,Math.PI*2);ctx.stroke();}if(document.getElementById('labels').checked){ctx.fillStyle='#ffffff';ctx.font='12px system-ui';let s=data.speed[i][j];ctx.fillText((data.displayLabels[String(data.ids[j])]?.display ?? ('Player '+data.ids[j]))+' / '+(s===null?'—':(s*3.6).toFixed(1)),X(p[0])+10,Y(p[1])-5);}}
let b=data.ball[i];if(b[0]!==null&&b[1]!==null){ctx.fillStyle='#fff';ctx.strokeStyle='#111';ctx.beginPath();ctx.arc(X(b[0]),Y(b[1]),6,0,Math.PI*2);ctx.fill();ctx.stroke();}let speed=data.ballSpeed[i];document.getElementById('state').textContent=t.toFixed(2)+'s | '+data.states[i]+' | owner '+(data.owner[i]===null?'none':(data.displayLabels[String(data.owner[i])]?.display??('Player '+data.owner[i])))+' | ball '+(speed===null?'unavailable':(speed*3.6).toFixed(1)+' km/h (estimated 2D)');updatePossession(i);updateIntelligence(t);updateTactics(t);}
const commentaryMode=document.getElementById('commentary-mode'),commentaryList=document.getElementById('commentary-list');
function renderCommentary(){commentaryList.replaceChildren();const items=data.intelligence.commentary.items.filter(item=>item.mode===commentaryMode.value);for(const item of items){const button=document.createElement('button');button.className='commentary-item';button.dataset.time=String(item.timestamp);button.style.display='block';button.style.margin='8px 0';button.textContent=item.timestamp.toFixed(2)+'s · '+item.mode+' · '+item.event_types.map(type=>type.replace('PASS','TRANSFER')).join(', ')+' — '+item.commentary;button.addEventListener('click',()=>seek(item.timestamp));commentaryList.appendChild(button);}const provider=data.intelligence.commentary.provider.provider;document.getElementById('commentary-status').textContent=provider==='none'?'Commentary disabled. Enable a provider when generating this report.':provider==='template'?'Local deterministic preview; no AI or network calls.':'Gemini editorial selection from supported facts.';if(data.intelligence.commentary.errors.length)document.getElementById('commentary-status').textContent+=' Some commentary requests failed or were rejected.';}
function updateIntelligence(t){let nearest=null,distance=Infinity;for(const button of commentaryList.children){button.style.outline='';button.removeAttribute('aria-current');const delta=Math.abs(Number(button.dataset.time)-t);if(delta<distance){distance=delta;nearest=button;}}if(nearest){nearest.style.outline='2px solid #1558b0';nearest.setAttribute('aria-current','true');}const callouts=document.getElementById('live-callouts');callouts.textContent='';if(document.getElementById('callouts').checked){const active=data.intelligence.events.events.filter(e=>['PLAYER_SPEED_THRESHOLD','BALL_SPEED_THRESHOLD','PLAYER_DISTANCE_MILESTONE'].includes(e.type)&&e.timestamp<=t&&t-e.timestamp<3);callouts.textContent=active.map(e=>e.type==='PLAYER_DISTANCE_MILESTONE'?('Player '+e.player_id+' measured distance ≥ '+e.metrics.milestone_m+' m (partial coverage)'):((e.player_id===undefined?'Ball (interpolated 2D)':'Player '+e.player_id)+' estimated speed '+e.metrics.speed_kmh.toFixed(1)+' km/h')).join(' · ');}}
commentaryMode.addEventListener('change',()=>{renderCommentary();updateIntelligence(Number(slider.value));});document.getElementById('callouts').addEventListener('change',()=>updateIntelligence(Number(slider.value)));renderCommentary();
function seek(t){slider.value=t;if(video)video.currentTime=t;draw(t);}
'''
    page += tactics_script()
    page += ask_script()
    page += '''slider.addEventListener('input',()=>seek(Number(slider.value)));document.getElementById('labels').addEventListener('change',()=>draw(Number(slider.value)));document.querySelectorAll('[data-seek]').forEach(b=>b.addEventListener('click',()=>seek(Number(b.dataset.seek))));if(video)video.addEventListener('timeupdate',()=>{slider.value=video.currentTime;draw(video.currentTime);});draw(0);</script></html>'''
    (output/'report.html').write_text(page)
