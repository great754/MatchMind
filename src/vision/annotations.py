"""Load half-video ball/action timestamps and align them to the MOT clip."""
import json

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

ACTION_CLASSES = ('Pass', 'Drive', 'Header', 'High Pass', 'Out', 'Cross',
                  'Throw In', 'Shot', 'Ball Player Block',
                  'Player Successful Tackle', 'Free Kick', 'Goal')
ACTION_COLORS = ((70,70,255), (30,150,255), (40,210,255), (70,240,180),
                 (90,220,90), (200,220,40), (220,180,40), (255,100,60),
                 (240,90,160), (220,60,220), (170,100,255), (100,180,255))


class ClipAnnotations:
    def __init__(self, paths, transformer, frame_count, fps):
        self.sync = json.loads((paths.root/'mot'/f'{paths.match_id}_sync.json').read_text())
        if self.sync['fps'] != fps:
            raise ValueError('Clip FPS differs from synchronization metadata')
        self.offset = int(self.sync['clip_start_frame_zero_based'])
        self.half = int(self.sync['half'])
        filename = paths.ball_first_half if self.half == 1 else paths.ball_second_half
        with np.load(filename) as data:
            frames = data['frame'].astype(int)
            if not np.array_equal(frames, np.arange(1, len(frames)+1)):
                raise ValueError('Expected consecutive one-based ball frames')
            if self.offset < 0 or self.offset+frame_count > len(frames):
                raise ValueError('Ball timeline does not cover clip')
            self.ball = np.column_stack((data['x']+52.5, data['y']+34))[self.offset:self.offset+frame_count]
            self.status = data['status'][self.offset:self.offset+frame_count].copy()
        self.project = LinearNDInterpolator(transformer.pitch_points, transformer.image_points)
        source = json.loads((paths.root/'bas'/paths.match_id/f'{paths.match_id}_12_class_events.json').read_text())
        if source['fps'] != fps or 'first frame' not in source.get('clock',''):
            raise ValueError('Action timestamps must use the released half-video clock')
        self.events = []
        for event in source['actions']:
            half = int(event['gameTime'].split('-')[0].strip())
            frame = int(event['frame'])-1-self.offset
            if half == self.half and 0 <= frame < frame_count:
                label = event['label'].title()
                if label not in ACTION_CLASSES:
                    raise ValueError(f'Unknown action: {label}')
                self.events.append({**event, 'label':label, 'clip_frame':frame,
                                    'clip_seconds':frame/fps,
                                    'half_video_seconds':float(event['position'])/1000})
        self.events.sort(key=lambda e:e['clip_frame'])
        self.actor_tracks = {str(p['player_id']): int(t) for t,p in self.sync['players'].items()}
        self.teams = {int(t):int(self.sync['side_to_color'][str(p['team'])])
                      for t,p in self.sync['players'].items()}
        self.fps = fps

    def ball_position(self, frame):
        p = self.ball[frame]
        # Unknown status does not provide a reliable ball position.
        return p if self.status[frame] != 0 and np.isfinite(p).all() else None

    def active_events(self, frame, hold_seconds=1.0):
        return [e for e in self.events if 0 <= frame-e['clip_frame'] < self.fps*hold_seconds]

    def draw(self, frame, frame_number):
        active = self.active_events(frame_number)
        height,width = frame.shape[:2]
        panel_w = min(730, round(width*0.20))
        row_h = max(24, round(height*0.074))
        x = width-panel_w-14
        panel = frame[8:8+12*row_h, x:width-14]
        panel[:] = cv2.addWeighted(panel,0.40,np.zeros_like(panel),0.60,0)
        labels={e['label'] for e in active}
        for i,(label,color) in enumerate(zip(ACTION_CLASSES,ACTION_COLORS)):
            y=8+i*row_h
            if label in labels:
                cv2.rectangle(frame,(x,y),(width-14,y+row_h),color,-1)
            cv2.rectangle(frame,(x,y),(width-14,y+row_h),color,2)
            text_color=(10,10,10) if label in labels else (220,220,220)
            scale=min(1.15,(panel_w-24)/max(1,cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]))
            cv2.putText(frame,label,(x+10,y+round(row_h*.66)),cv2.FONT_HERSHEY_SIMPLEX,
                        scale,text_color,2,cv2.LINE_AA)
        if active:
            e=active[-1];side=0 if e['team']=='left' else 1
            team=int(self.sync['side_to_color'][str(side)])
            color=(255,80,80) if team==0 else (80,80,255)
            cv2.putText(frame,f"Player: {e['player_id']} ({e['team']})",(20,150),
                        cv2.FONT_HERSHEY_SIMPLEX,0.85,color,2,cv2.LINE_AA)
            track=self.actor_tracks.get(str(e['player_id']))
            return track
        return None

    def draw_projected_ball(self, frame, number):
        ball=self.ball_position(number)
        if ball is None:return
        point=np.asarray(self.project(ball)).reshape(-1)
        if not np.isfinite(point).all():return
        x,y=map(lambda v:int(round(v)),point)
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):return
        cv2.circle(frame,(x,y),15,(0,0,0),4,cv2.LINE_AA)
        cv2.circle(frame,(x,y),15,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(frame,(x,y),4,(255,255,255),-1,cv2.LINE_AA)
