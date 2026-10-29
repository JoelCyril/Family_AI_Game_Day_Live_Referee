"""Local MVP server. Do not expose on public internet without access controls."""
import json
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from engine import Engine, GAMES, verify_report

app=FastAPI(title='Family AI Game Day — Live Referee MVP')
ROOT=Path(__file__).parent
app.mount('/static',StaticFiles(directory=str(ROOT/'static')),name='static')
sessions={}

@app.get('/')
def index():
    return FileResponse(ROOT/'static'/'index.html')

@app.get('/api/games')
def games():
    return GAMES

@app.get('/api/report/{session_id}')
def report(session_id:str):
    engine=sessions.get(session_id)
    if not engine:raise HTTPException(status_code=404,detail='Session not found')
    return engine.report()

@app.post('/api/verify')
def verify(report:dict=Body(...)):
    valid,description=verify_report(report)
    return {'valid':valid,'description':description}

@app.websocket('/ws')
async def ws_referee(ws:WebSocket):
    await ws.accept()
    engine=Engine()
    sessions[engine.id]=engine
    latest_frame=None
    await ws.send_json({'type':'hello','state':engine.public_state()})
    try:
        while True:
            msg=await ws.receive()
            if msg.get('type')=='websocket.disconnect':break
            if msg.get('bytes') is not None:
                buf=msg['bytes']
                if len(buf)>2_000_000:
                    await ws.send_json({'type':'error','message':'Frame too large'});continue
                encoded=np.frombuffer(buf,np.uint8)
                latest_frame=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                if latest_frame is None:
                    await ws.send_json({'type':'error','message':'Could not decode JPEG'});continue
                h,w=latest_frame.shape[:2]
                if w>960 or h>960:
                    latest_frame=cv2.resize(latest_frame,(min(w,960),min(h,540)))
                try:
                    outcome=engine.process(latest_frame)
                    await ws.send_json(outcome)
                except Exception as ex:
                    await ws.send_json({'type':'error','message':f'Frame processing failed: {ex}'})
            elif msg.get('text') is not None:
                try:
                    cmd=json.loads(msg['text'])
                    state=engine.control(cmd,latest_frame)
                    await ws.send_json({'type':'control','state':state})
                except (ValueError, KeyError, TypeError) as ex:
                    await ws.send_json({'type':'error','message':str(ex)})
    except WebSocketDisconnect:
        pass
    finally:
        engine.close()
        # Session events are retained in memory for report download while server is running.
