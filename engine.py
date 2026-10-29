"""Rule-driven scorer for Family AI Game Day (ADVANCED prototype).

Rules are invented competition adaptations, NOT official heritage rules.
For false-positive prevention, requires transitions and debouncing.
Sagala candidates are never automatically awarded as verified points.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import os
import secrets
import json
import math
import time
import uuid
import cv2
import numpy as np
from vision import PRESETS, hsv_mask, largest_marker, largest_round_marker, coverage, in_zone, detect_blobs, PoseHands, OptionalYolo, detect_team_players, classify_person_team

SIGNING_KEY = (os.getenv('JUDGE_SIGNING_KEY') or secrets.token_hex(32)).encode('utf-8')

GAMES = {
    'football': {'name':'Football', 'zone':[.86,.34,.11,.32], 'hsv':'green', 'hint':'Aim camera at a small goal. Draw goal-scoring area. Use a bright green ball or enable YOLO tracking. Goal entry is a candidate, not line-crossing proof.'},
    'lagori': {'name':'Lagori — India', 'zone':[.36,.23,.28,.54], 'hsv':'green', 'hint':'Put a green-painted seven-piece stack on a contrasting dark mat. Draw the stack area. Click Capture baseline BEFORE playing.'},
    'kancha': {'name':'Kancha — India', 'zone':[.42,.36,.16,.25], 'hsv':'green', 'hint':'Top-down camera. Use a large bright green marble and draw the pocket/target area. A re-entry scores once.'},
    'gaheef': {'name':'Al Gaheef — UAE', 'zone':[.16,.1,.68,.82], 'hsv':'green', 'hint':'Draw the entire 3×3 hopping grid; number cells left-to-right, top-to-bottom. MediaPipe Pose is required to award hop scores.'},
    'sagala': {'name':'Al Sagala — UAE', 'zone':[.15,.12,.70,.80], 'hsv':'green', 'hint':'Close-up on hands and five large green pebbles on a dark mat. MediaPipe Hands detects candidate catches; manual confirmation needed.'},
}


def center(p):
    return (p['x'],p['y']) if p else None

def normzone(zone):
    if not isinstance(zone,(list,tuple)) or len(zone)!=4: raise ValueError('Zone must be [x, y, width, height]')
    x,y,w,h=[float(v) for v in zone]
    if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0: raise ValueError('Invalid zone')
    x=max(0,min(.99,x)); y=max(0,min(.99,y))
    return [round(x,4),round(y,4),round(min(w,1-x),4),round(min(h,1-y),4)]


class Engine:
    def __init__(self):
        self.id=uuid.uuid4().hex[:16]
        self.vision=PoseHands()
        self.yolo=OptionalYolo()
        self.scores={'A':0,'B':0}
        self.events=[]
        self.last_hash='0'*64
        self.frame_id=0
        self.matches=[]
        self.duration=60
        self.teams={'A':'Team A','B':'Team B'}
        self.players={'A':['P1','P2'],'B':['P3','P4']}
        self.team_colors={'A':'red','B':'blue'}
        self.ball_color='green'
        self.football_ball_mode='yolo'
        self.goal_calibrated=False
        self.team_possession={'A':0.0,'B':0.0}
        self.player_possession=defaultdict(float)
        self.last_possession=None
        self.active_possession_player=None
        self.started=False
        self.last_frame_at=None
        self.elapsed=0.0
        self.turn='A'
        self.set_game('football',reset_session=False)
        self.model_start_ms=int(time.monotonic()*1000)

    def set_game(self,key,reset_session=True):
        if key not in GAMES: raise ValueError('Unknown game')
        if reset_session and getattr(self,'events',None):
            self.matches.append({'game':self.game,'scores':dict(self.scores),
                'events':list(self.events), 'last_hash':self.last_hash,
                'winner':('Draw' if self.scores['A']==self.scores['B'] else
                          ('A' if self.scores['A']>self.scores['B'] else 'B'))})
        self.game=key
        self.zone=list(GAMES[key]['zone'])
        self.goal_calibrated=(key!='football')  # Football requires explicit calibration or preset to prevent window goals
        self.football_ball_mode='yolo'
        self.hsv=PRESETS[GAMES[key]['hsv']]
        self.scores={'A':0,'B':0}
        self.events=[]
        self.last_hash='0'*64
        self.started=False
        self.finished=False
        self.turn='A'
        self.elapsed=0.0
        self.last_frame_at=None
        self.previous_inside=False
        self.missing_frames=0
        self.inside_frames=0
        self.armed=False
        self.last_scored_at=0.0
        self.stack_baseline=None
        self.stack_phase='standing'
        self.stack_frames=0
        self.next_cell=0
        self.cell_streak=0
        self.last_cell=None
        self.previous_palm=None
        self.hand_raised_at=None
        self.hand_closed_before=False
        self.held_catch=None
        self.sagala_candidates=0
        self.possession_seconds=defaultdict(float)
        self.team_possession={'A':0.0,'B':0.0}
        self.player_possession=defaultdict(float)
        self.last_possession=None
        self.active_possession_player=None
        self._prev_now=None
        self.yolo_track_reset_needed=True

    def public_state(self):
        tot_pos = self.team_possession['A'] + self.team_possession['B']
        pct_a = round((self.team_possession['A']/tot_pos*100),1) if tot_pos>0 else 50.0
        pct_b = round((self.team_possession['B']/tot_pos*100),1) if tot_pos>0 else 50.0
        mvp = None
        if self.player_possession:
            top_p, top_sec = max(self.player_possession.items(), key=lambda it:it[1])
            top_team = 'A' if top_p in self.players['A'] else ('B' if top_p in self.players['B'] else 'A')
            mvp = {'name':top_p,'team':self.teams[top_team],'seconds':round(top_sec,1)}
        top_players = sorted([{'player':k,'seconds':round(v,1),'team':('A' if k in self.players['A'] else 'B')} for k,v in self.player_possession.items()], key=lambda it:-it['seconds'])
        top_yolo = sorted(self.possession_seconds.items(),key=lambda it:-it[1])
        return {'session':self.id,'game':self.game,'name':GAMES[self.game]['name'],'zone':self.zone,
                'scores':self.scores,'started':self.started,'finished':self.finished,
                'winner':(None if not self.finished else ('Draw' if self.scores['A']==self.scores['B'] else self.teams['A' if self.scores['A']>self.scores['B'] else 'B'])),
                'turn':self.turn,'elapsed':round(self.elapsed,1),
                'duration':self.duration,'teams':self.teams,'players':self.players,'matches':len(self.matches),
                'team_colors':self.team_colors,'ball_color':self.ball_color,
                'football_ball_mode':self.football_ball_mode,'goal_calibrated':self.goal_calibrated,
                'team_possession':{'A':round(self.team_possession['A'],1),'B':round(self.team_possession['B'],1)},
                'possession_pct':{'A':pct_a,'B':pct_b},
                'player_possession':top_players[:6],
                'mvp':mvp,
                'active_possession':self.active_possession_player,
                'next_cell':self.next_cell+1 if self.next_cell<9 else 'complete',
                'baseline':round(self.stack_baseline,4) if self.stack_baseline is not None else None,
                'sagala_candidates':self.sagala_candidates,
                'possession':[{'player_id':k,'seconds':round(v,1)} for k,v in top_yolo[:5]],
                'models':{'pose':self.vision.pose_reason,'hands':self.vision.hands_reason,'yolo':self.yolo.reason},
                'events':list(reversed(self.events[-12:])), 'event_count':len(self.events),
                'hsv':[int(v) for v in self.hsv]}

    def add_event(self,kind,label,points=0,confidence=None,review=False,evidence=None,team=None):
        target_team = team if team in ('A','B') else self.turn
        evt={'number':len(self.events)+1,'timestamp':datetime.now(timezone.utc).isoformat(),
             'game':self.game,'turn':target_team,'kind':kind,'label':label,'points':int(points),
             'confidence':confidence,'review_required':bool(review),'evidence':evidence or {},
             'prev_hash':self.last_hash}
        payload=json.dumps(evt,sort_keys=True,separators=(',',':')).encode()
        evt['hash']=sha256(payload).hexdigest()
        evt['signature']=hmac.new(SIGNING_KEY,evt['hash'].encode(),sha256).hexdigest()
        self.last_hash=evt['hash']
        self.events.append(evt)
        if points: self.scores[target_team]+=int(points)
        return evt

    def control(self,msg,latest_frame=None):
        cmd=msg.get('command')
        if cmd=='game': self.set_game(str(msg['value']))
        elif cmd=='start':
            if self.finished: raise ValueError('Round has ended; use Reset to start a new round')
            self.started=True; self.last_frame_at=None
        elif cmd=='stop': self.started=False; self.last_frame_at=None
        elif cmd=='finish':
            self.started=False
            if not self.finished:
                self.finished=True
                self.add_event('round_end','Host ended the round; winner calculated',0)
        elif cmd=='reset': self.set_game(self.game)
        elif cmd=='turn': self.turn = 'A' if msg.get('value')=='A' else 'B'; self.previous_inside=False;self.armed=False
        elif cmd=='team':
            team=str(msg.get('team','A'))
            if team not in ('A','B'):raise ValueError('Unknown team')
            self.teams[team]=str(msg.get('name',''))[:32] or ('Team '+team)
            self.players[team]=[str(x)[:24] for x in msg.get('players',[])[:10]]
        elif cmd=='duration':
            self.duration=max(15,min(3600,int(msg['value'])))
        elif cmd=='zone':
            self.zone=normzone(msg['value'])
            self.goal_calibrated=True
            self.previous_inside=False;self.armed=False
        elif cmd=='default_goal':
            self.zone=list(GAMES['football']['zone'])
            self.goal_calibrated=True
            self.previous_inside=False;self.armed=False
        elif cmd=='ball_mode':
            m=str(msg.get('value','marker'))
            if m in ('yolo','marker'):
                self.football_ball_mode=m
        elif cmd=='color':
            name=str(msg['value'])
            if name not in PRESETS:raise ValueError('Unsupported preset')
            self.hsv=PRESETS[name]
        elif cmd=='team_color':
            t=str(msg.get('team','A'))
            if t not in ('A','B'):raise ValueError('Unknown team')
            c=str(msg.get('value','red'))
            if c not in PRESETS:raise ValueError('Unsupported color preset')
            self.team_colors[t]=c
        elif cmd=='ball_color':
            c=str(msg.get('value','green'))
            if c not in PRESETS:raise ValueError('Unsupported color preset')
            self.ball_color=c
        elif cmd=='sample':
            if latest_frame is None:raise ValueError('No frame to sample yet')
            x,y=(float(v) for v in msg['value'])
            h,w=latest_frame.shape[:2]
            px,py=max(0,min(w-1,int(x*w))),max(0,min(h-1,int(y*h)))
            patch=latest_frame[max(0,py-3):min(h,py+4),max(0,px-3):min(w,px+4)]
            med=np.median(patch.reshape(-1,3),axis=0).astype(np.uint8)
            self.hsv=tuple(map(int,cv2.cvtColor(med.reshape(1,1,3),cv2.COLOR_BGR2HSV)[0,0]))
        elif cmd=='baseline':
            if self.game!='lagori' or latest_frame is None:raise ValueError('Show the stack first in Lagori')
            self.stack_baseline=coverage(latest_frame,self.hsv,self.zone)
            if self.stack_baseline<.005:raise ValueError('Not enough selected color detected in stack zone')
            self.stack_phase='standing'; self.stack_frames=0
        elif cmd=='manual_score':
            points=int(msg.get('points',1))
            if points not in (-1,1):raise ValueError('Allowed corrections: +1 or -1')
            if self.scores[self.turn]+points<0:raise ValueError('Score cannot be negative')
            self.add_event('manual','Host adjustment '+('+' if points>0 else '')+str(points),points,review=True)
        elif cmd=='audio_cue':
            words=str(msg.get('value',''))[:150]
            if words.strip() and self.started:
                self.add_event('audio_cue','Audio phrase detected: '+words,0,
                               confidence='speech recognition',review=True)
        elif cmd=='confirm_sagala':
            if self.game!='sagala' or not self.sagala_candidates:raise ValueError('No candidate to confirm')
            self.sagala_candidates-=1
            self.add_event('confirmed_catch','Host confirmed observed Sagala catch',1,review=False)
        elif cmd=='reject_sagala':
            if self.game!='sagala' or not self.sagala_candidates:raise ValueError('No candidate to reject')
            self.sagala_candidates-=1
            self.add_event('rejected_catch','Host rejected uncertain Sagala catch',0,review=False)
        elif cmd=='enable_yolo':
            self.yolo.enable()
        else:raise ValueError('Unknown command')
        return self.public_state()

    def process(self,frame:np.ndarray,now=None):
        if frame is None or frame.size==0:raise ValueError('Invalid frame')
        now=time.monotonic() if now is None else now
        self.frame_id+=1
        if self.started and self.last_frame_at is not None:
            self.elapsed+=max(0,now-self.last_frame_at)
        self.last_frame_at=now
        if self.started and self.elapsed>=self.duration:
            self.started=False
            self.finished=True
            self.add_event('time_up','Round timer finished; scores frozen',0,review=False)
        dots=[]
        diag={}
        if self.game=='football': dots,diag=self._football(frame,now)
        elif self.game=='lagori': dots,diag=self._lagori(frame,now)
        elif self.game=='kancha': dots,diag=self._kancha(frame,now)
        elif self.game=='gaheef': dots,diag=self._gaheef(frame,now)
        elif self.game=='sagala': dots,diag=self._sagala(frame,now)
        return {'type':'frame','state':self.public_state(),'detections':dots,'diagnostics':diag}

    def _track_zone_entry(self,pt,now,required_frames=2,cooldown=1.5):
        # Must see target outside at least once after reset or last hit.
        inside=in_zone(pt,self.zone)
        if not inside:
            if pt is not None:self.armed=True
            self.inside_frames=0
        else:
            self.inside_frames+=1
        scored=(self.armed and inside and self.inside_frames==required_frames and now-self.last_scored_at>=cooldown)
        if scored:
            self.last_scored_at=now
            self.armed=False
        self.previous_inside=inside
        return scored

    def _football(self,frame,now):
        hsv_ball = PRESETS.get(self.ball_color, self.hsv)
        hsv_a = PRESETS.get(self.team_colors.get('A','red'), (0,210,190))
        hsv_b = PRESETS.get(self.team_colors.get('B','blue'), (115,210,190))

        marker = None
        people, model_ball = self.yolo.analyze(frame)
        if self.football_ball_mode == 'yolo' and self.yolo.model is not None:
            ball = model_ball
            ball_source = 'YOLO sports ball' if ball else 'not detected'
        else:
            # Strict circular marker check (requires roundness >= 0.60, rejects flat wall paintings)
            marker, _ = largest_round_marker(frame, hsv_ball)
            ball = marker if marker else model_ball
            ball_source = f"circular marker ({self.ball_color})" if marker else ('YOLO ball' if model_ball else 'not detected')
        pt = center(ball)

        all_players=[]
        if self.yolo.model is not None and people:
            for p in people:
                team_guess = classify_person_team(frame, p['box'], hsv_a, hsv_b)
                if not team_guess:
                    team_guess = 'A' if p['box'][0] < 0.5 else 'B'
                names = self.players.get(team_guess, ['P1'])
                p_name = names[(p['id'] % len(names))] if names else f"{team_guess}{p['id']}"
                foot = [p['box'][0]+p['box'][2]/2, min(1.0, p['box'][1]+p['box'][3]*0.92)]
                all_players.append({
                    'team': team_guess,
                    'player': p_name,
                    'box': p['box'],
                    'x': p['box'][0]+p['box'][2]/2,
                    'y': p['box'][1]+p['box'][3]/2,
                    'foot': foot,
                    'id': p['id']
                })
        else:
            # Color-based jersey tracking for Team A
            dets_a = detect_team_players(frame, hsv_a, min_area=60)
            names_a = self.players.get('A', ['P1', 'P2'])
            for i, p in enumerate(dets_a):
                p_name = names_a[i % len(names_a)] if names_a else f"A{i+1}"
                all_players.append({
                    'team': 'A',
                    'player': p_name,
                    'box': p['box'],
                    'x': p['x'],
                    'y': p['y'],
                    'foot': p['foot'],
                    'id': i+1
                })
            # Color-based jersey tracking for Team B
            dets_b = detect_team_players(frame, hsv_b, min_area=60)
            names_b = self.players.get('B', ['P3', 'P4'])
            for i, p in enumerate(dets_b):
                p_name = names_b[i % len(names_b)] if names_b else f"B{i+1}"
                all_players.append({
                    'team': 'B',
                    'player': p_name,
                    'box': p['box'],
                    'x': p['x'],
                    'y': p['y'],
                    'foot': p['foot'],
                    'id': i+1
                })

        # Calculate live possession & MVP
        active_p = None
        if self.started and pt and all_players:
            closest = min(all_players, key=lambda p: math.dist(pt, p['foot']))
            dist = math.dist(pt, closest['foot'])
            # Within 18% normalized frame distance to player's feet
            if dist < 0.18:
                sec = max(0.0, min(0.35, now - (self._prev_now or now)))
                self.team_possession[closest['team']] += sec
                self.player_possession[closest['player']] += sec
                self.last_possession = {'team': closest['team'], 'player': closest['player'], 'at': now}
                active_p = closest['player']
        self._prev_now = now
        self.active_possession_player = active_p

        # Goal detection (requires goal_calibrated to prevent phantom goals on windows)
        if self.goal_calibrated and self.started and self._track_zone_entry(pt, now, required_frames=2, cooldown=2.0):
            # Attribute to recently possessing team, or currently attacking team
            if self.last_possession and (now - self.last_possession.get('at', 0) < 4.0):
                scoring_team = self.last_possession['team']
                scorer = self.last_possession['player']
            else:
                scoring_team = self.turn
                scorer = f"{self.teams[scoring_team]} Strike"
            self.add_event('goal', f"GOAL! {self.teams[scoring_team]} scores! Scorer: {scorer}", points=1,
                           confidence='confirmed', review=False, team=scoring_team,
                           evidence={'ball': pt, 'scorer': scorer, 'team': scoring_team, 'zone': self.zone})

        dots = []
        if ball:
            dots.append({'kind':'ball', 'x':ball['x'], 'y':ball['y'], 'label':'⚽ BALL'})
        for p in all_players:
            label = f"{p['player']} ({p['team']})"
            if p['player'] == active_p:
                label += " [POSSESSION]"
            dots.append({
                'kind': 'player',
                'team': p['team'],
                'x': p['x'],
                'y': p['y'],
                'foot': p['foot'],
                'box': p['box'],
                'label': label,
                'has_ball': (p['player'] == active_p)
            })

        top_p = max(self.player_possession.items(), key=lambda it: it[1])[0] if self.player_possession else '—'
        diag = {
            'ball_source': 'color ('+self.ball_color+')' if marker else ('YOLO' if model_ball else 'not detected'),
            'team_a_players': len([p for p in all_players if p['team']=='A']),
            'team_b_players': len([p for p in all_players if p['team']=='B']),
            'team_possession_seconds': {'A': round(self.team_possession['A'],1), 'B': round(self.team_possession['B'],1)},
            'mvp_leader': top_p,
            'active_dribbler': active_p or 'contested / loose ball'
        }
        return dots, diag

    def _lagori(self,frame,now):
        cov=coverage(frame,self.hsv,self.zone)
        base=self.stack_baseline
        if self.started and base:
            ratio=cov/max(.001,base)
            if self.stack_phase=='standing':
                self.stack_frames=self.stack_frames+1 if ratio<.55 else 0
                if self.stack_frames>=3:
                    self.stack_phase='fallen';self.stack_frames=0
                    self.add_event('stack_fallen','Stack changed from standing to fallen',1,
                                    confidence='medium',review=True,evidence={'ratio':round(ratio,3)})
            else:
                self.stack_frames=self.stack_frames+1 if ratio>.82 else 0
                if self.stack_frames>=3:
                    self.stack_phase='standing';self.stack_frames=0
                    self.add_event('stack_rebuilt','Stack restored to baseline',2,
                                    confidence='medium',review=True,evidence={'ratio':round(ratio,3)})
        return [],{'coverage':round(cov,4),'baseline':base,'phase':self.stack_phase,
                   'method':'colored stack occupancy change, not ball-hit proof'}

    def _kancha(self,frame,now):
        marker,_=largest_marker(frame,self.hsv,min_area=7)
        pt=center(marker)
        if self.started and self._track_zone_entry(pt,now,required_frames=2,cooldown=1.0):
            self.add_event('target_entry','Marble entered marked target region',1,
                           confidence='medium',review=True,evidence={'marble':pt,'zone':self.zone})
        return ([{'kind':'marble','x':pt[0],'y':pt[1],'label':'MARBLE'}] if pt else []),{'detected':bool(marker),'inside_target':in_zone(pt,self.zone)}

    def _gaheef(self,frame,now):
        if self.vision.pose is None:
            return [],{'status':'Pose AI not loaded; no scores will be awarded','model':self.vision.pose_reason}
        landmarks=self.vision.landmarks(frame,int(now*1000),'pose')
        if not landmarks or len(landmarks[0])<33:
            self.cell_streak=0
            return [],{'status':'Full-body pose not detected'}
        feet=[landmarks[0][31],landmarks[0][32]]
        feet=[f for f in feet if f['visibility']>=.5 and 0<=f['x']<=1 and 0<=f['y']<=1]
        dots=[{'kind':'foot','x':f['x'],'y':f['y'],'label':'FOOT'} for f in feet]
        x,y,w,h=self.zone
        cells=[]
        for f in feet:
            if in_zone((f['x'],f['y']),self.zone):
                col=min(2,max(0,int((f['x']-x)/w*3)))
                row=min(2,max(0,int((f['y']-y)/h*3)))
                cells.append(row*3+col)
        expected=self.next_cell
        candidate = expected if expected in cells else None
        if not self.started or expected>=9:return dots,{'observed_cells':[n+1 for n in cells],'expected_cell':expected+1}
        self.cell_streak = self.cell_streak+1 if candidate is not None else 0
        if self.cell_streak>=3 and now-self.last_scored_at>.4:
            self.last_scored_at=now
            self.next_cell+=1
            self.cell_streak=0
            self.add_event('hop_step',f'Foot entered correct grid cell {expected+1}',1,
                           confidence='medium',review=True,evidence={'cell':expected+1,'foot_count':len(feet)})
            if self.next_cell==9:
                self.add_event('grid_complete','Completed the 9-cell route',3,confidence='medium',review=True)
        return dots,{'observed_cells':[n+1 for n in cells],'expected_cell':min(9,self.next_cell+1),
                     'warning':'Foot position in 2-D does not prove foot-ground contact or a valid hop'}

    def _sagala(self,frame,now):
        pebbles=detect_blobs(frame,self.hsv,self.zone,min_area=5)
        dots=[{'kind':'stone','x':p['x'],'y':p['y'],'label':'PEBBLE'} for p in pebbles[:8]]
        if self.vision.hands is None:
            return dots,{'status':'Hand AI not loaded; no catch candidates can be detected','stones_visible':len(pebbles)}
        all_hands=self.vision.landmarks(frame,int(now*1000),'hands')
        if not all_hands: return dots,{'status':'Hand not detected','stones_visible':len(pebbles)}
        hand=all_hands[0]
        palm=(hand[0]['x'],hand[0]['y'])
        # Finger-curl heuristic based on ML-landmark geometry, not a verified catch.
        palm_center=((hand[0]['x']+hand[9]['x'])/2,(hand[0]['y']+hand[9]['y'])/2)
        fold=0
        for tip,base in [(8,5),(12,9),(16,13),(20,17)]:
            if math.dist((hand[tip]['x'],hand[tip]['y']),palm_center)<1.15*math.dist((hand[base]['x'],hand[base]['y']),palm_center):
                fold+=1
        closed=fold>=3
        dots.append({'kind':'hand','x':palm[0],'y':palm[1],'label':'FIST' if closed else 'HAND'})
        near_pebble=any(math.dist(palm,(p['x'],p['y']))<.18 for p in pebbles)
        if self.started and self.previous_palm is not None and self.previous_palm[1]-palm[1]>.045:
            self.hand_raised_at=now
        # Catch candidate only when raised hand goes from open to closed near stone.
        if (self.started and closed and not self.hand_closed_before
                and self.hand_raised_at is not None and 0<now-self.hand_raised_at<1.6
                and near_pebble and now-self.last_scored_at>1.1):
            self.last_scored_at=now
            self.sagala_candidates+=1
            self.add_event('candidate_catch','Possible catch: hand closed near colored pebble',0,
                           confidence='low',review=True,evidence={'palm':palm,'near_pebble':True})
            self.hand_raised_at=None
        self.hand_closed_before=closed
        self.previous_palm=palm
        return dots,{'visible_pebbles':len(pebbles),'hand_closed':closed,'pending_review':self.sagala_candidates,
                     'warning':'Candidate only. A genuine throw-and-catch cannot be confirmed from this heuristic.'}

    def close(self):
        self.vision.close()

    def report(self):
        tot_pos = self.team_possession['A'] + self.team_possession['B']
        pct_a = round((self.team_possession['A']/tot_pos*100),1) if tot_pos>0 else 50.0
        pct_b = round((self.team_possession['B']/tot_pos*100),1) if tot_pos>0 else 50.0
        mvp = None
        if self.player_possession:
            top_p, top_sec = max(self.player_possession.items(), key=lambda it:it[1])
            top_team = 'A' if top_p in self.players['A'] else ('B' if top_p in self.players['B'] else 'A')
            mvp = {'name':top_p,'team':self.teams[top_team],'seconds':round(top_sec,1)}
        return {'title':'Family AI Game Day — Live Referee Report', 'session_id':self.id,
                'generated_at':datetime.now(timezone.utc).isoformat(),
                'game':self.game, 'scores':self.scores,
                'team_possession':{'A':round(self.team_possession['A'],1),'B':round(self.team_possession['B'],1)},
                'possession_pct':{'A':pct_a,'B':pct_b},
                'player_possession':dict(self.player_possession),
                'mvp':mvp,
                'events':self.events,'archived_matches':self.matches,'final_hash':self.last_hash,
                'methodology':'Hash chained server-generated rule events with HMAC signature verification.'}


def verify_report(report:dict) -> tuple[bool,str]:
    """Verify integrity/authenticity of locally generated events using a private server key."""
    games=list(report.get('archived_matches',[]))+[{'events':report.get('events',[]),'last_hash':report.get('final_hash','0'*64)}]
    for round_index, game in enumerate(games):
        previous='0'*64
        for e in game.get('events',[]):
            item={k:v for k,v in e.items() if k not in ('hash','signature')}
            if item.get('prev_hash')!=previous:return False,f'Round {round_index}: previous event hash mismatch'
            value=sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if value!=e.get('hash'):return False,f'Round {round_index}: event data was changed'
            sig=hmac.new(SIGNING_KEY,value.encode(),sha256).hexdigest()
            if not hmac.compare_digest(sig,str(e.get('signature',''))):return False,f'Round {round_index}: signature failed'
            previous=value
        if previous!=game.get('last_hash',report.get('final_hash')):return False,'Incorrect final hash'
    return True,'All events verified with the local referee server key'
