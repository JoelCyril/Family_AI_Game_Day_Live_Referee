from fastapi.testclient import TestClient
from app import app


def test_http_and_websocket():
    client=TestClient(app)
    assert client.get('/').status_code==200
    assert len(client.get('/api/games').json())==5
    with client.websocket_connect('/ws') as socket:
        hello=socket.receive_json()
        assert hello['type']=='hello'
        sid=hello['state']['session']
        socket.send_json({'command':'game','value':'kancha'})
        response=socket.receive_json()
        assert response['state']['game']=='kancha'
        report=client.get(f'/api/report/{sid}').json()
        assert client.post('/api/verify',json=report).json()['valid']


def test_websocket_live_jpeg_scoring():
    import cv2
    import numpy as np
    frame=np.zeros((360,640,3),np.uint8)
    cv2.circle(frame,(10,100),15,(0,255,0),-1)
    inside=np.zeros_like(frame)
    cv2.circle(inside,(300,170),15,(0,255,0),-1)
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as socket:
            socket.receive_json()
            for cmd in ({'command':'game','value':'kancha'},{'command':'start'}):
                socket.send_json(cmd);socket.receive_json()
            for fr in (frame,inside,inside):
                ok,jpeg=cv2.imencode('.jpg',fr)
                assert ok
                socket.send_bytes(jpeg.tobytes())
                response=socket.receive_json()
                assert response['type']=='frame'
            assert response['state']['scores']['A']==1
            report=client.get('/api/report/'+response['state']['session']).json()
            assert client.post('/api/verify',json=report).json()['valid']


def test_football_full_flow_via_websocket():
    import cv2
    import numpy as np
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as socket:
            socket.receive_json() # hello
            socket.send_json({'command':'game','value':'football'})
            st = socket.receive_json()['state']
            assert st['game'] == 'football'

            socket.send_json({'command':'default_goal'})
            socket.receive_json()

            socket.send_json({'command':'team','team':'A','name':'Falcons','players':['Ali','Rashid']})
            socket.receive_json()
            socket.send_json({'command':'team','team':'B','name':'Oryx','players':['Zayed','Omar']})
            socket.receive_json()
            socket.send_json({'command':'start'})
            socket.receive_json()

            # Send frames: Team A player at (120, 160) with ball at (140, 180)
            import time
            frame_pos = np.zeros((360, 640, 3), np.uint8)
            cv2.rectangle(frame_pos, (100, 120), (140, 190), (0, 0, 220), -1) # Red jersey
            cv2.circle(frame_pos, (135, 195), 8, (0, 255, 0), -1) # Green ball
            ok, jpeg = cv2.imencode('.jpg', frame_pos)
            for _ in range(4):
                time.sleep(0.06)
                socket.send_bytes(jpeg.tobytes())
                resp = socket.receive_json()

            assert resp['state']['team_possession']['A'] > 0
            assert resp['state']['mvp'] is not None
            assert resp['state']['mvp']['name'] == 'Ali'

            # Now send goal frame (ball inside goal zone at 580, 180)
            frame_goal = np.zeros((360, 640, 3), np.uint8)
            cv2.circle(frame_goal, (580, 180), 8, (0, 255, 0), -1)
            ok, jpeg_goal = cv2.imencode('.jpg', frame_goal)
            for _ in range(3):
                time.sleep(0.06)
                socket.send_bytes(jpeg_goal.tobytes())
                resp = socket.receive_json()

            assert resp['state']['scores']['A'] >= 1
            sid = resp['state']['session']
            report = client.get(f'/api/report/{sid}').json()
            assert report['mvp']['name'] == 'Ali'
            assert client.post('/api/verify', json=report).json()['valid']
