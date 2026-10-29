import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from engine import Engine, verify_report
from vision import hsv_mask, in_zone


def board(x=15,y=40):
    image=np.zeros((360,640,3),dtype=np.uint8)
    cv2.circle(image,(x,y),14,(0,255,0),-1)
    return image


def test_color_detector():
    image=board()
    mask=hsv_mask(image,(66,52,52))
    assert mask.sum()>0
    assert in_zone((.5,.5),[.4,.4,.2,.2])
    assert not in_zone((.1,.5),[.4,.4,.2,.2])


def test_kancha_does_not_score_stationary():
    e=Engine()
    e.set_game('kancha')
    e.control({'command':'start'})
    e.process(board(x=10,y=100),now=100)
    e.process(board(x=300,y=170),now=101)
    e.process(board(x=300,y=170),now=102)
    assert e.scores['A']==1,e.public_state()
    for i in range(10):e.process(board(x=300,y=170),now=103+i)
    assert e.scores['A']==1
    assert len(e.events)==1
    assert verify_report(e.report())[0]
    tampered=e.report()
    tampered['events'][0]['label']='fake'
    assert not verify_report(tampered)[0]
    e.close()


def test_football_goal_region():
    e=Engine()
    e.set_game('football')
    e.control({'command':'default_goal'})
    e.control({'command':'start'})
    e.process(board(x=20,y=165),now=100)
    e.process(board(x=590,y=165),now=101)
    e.process(board(x=590,y=165),now=102)
    assert e.scores['A']==1
    e.close()


def test_lagori_baseline_knockdown_rebuild():
    e=Engine();e.set_game('lagori')
    frame=np.zeros((360,640,3),np.uint8)
    cv2.rectangle(frame,(245,115),(315,245),(0,255,0),-1)
    e.control({'command':'baseline'},frame)
    e.control({'command':'start'})
    for t in range(3):e.process(np.zeros_like(frame),now=100+t)
    assert e.scores['A']==1
    for t in range(3):e.process(frame,now=104+t)
    assert e.scores['A']==3
    assert e.events[0]['kind']=='stack_fallen'
    e.close()


def test_match_archival_and_time_limit():
    e=Engine();e.set_game('kancha')
    e.control({'command':'duration','value':15})
    e.control({'command':'start'})
    e.process(board(x=10,y=100),now=100)
    e.process(board(x=300,y=170),now=101)
    e.process(board(x=300,y=170),now=102)
    e.control({'command':'game','value':'gaheef'})
    assert len(e.matches)==1
    assert e.matches[0]['scores']['A']==1
    assert verify_report(e.report())[0]
    e.close()


def test_football_team_possession_and_mvp():
    e = Engine()
    e.set_game('football')
    e.control({'command': 'team', 'team': 'A', 'name': 'Falcons', 'players': ['Ali', 'Rashid']})
    e.control({'command': 'team', 'team': 'B', 'name': 'Oryx', 'players': ['Zayed', 'Omar']})
    e.control({'command': 'start'})
    # Create a frame with Team A player (Red jersey: BGR = (0, 0, 220)) and neon green ball (BGR = (0, 255, 0))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Player A torso (Red)
    cv2.rectangle(frame, (100, 100), (140, 180), (0, 0, 220), -1)
    # Ball near player's feet (feet at ~ 120, 180)
    cv2.circle(frame, (130, 190), 8, (0, 255, 0), -1)
    # Process 5 consecutive frames
    for t in range(5):
        res = e.process(frame, now=100.0 + t * 0.1)
    st = e.public_state()
    assert st['team_possession']['A'] > 0.0
    assert st['team_possession']['B'] == 0.0
    assert st['possession_pct']['A'] == 100.0
    assert st['mvp'] is not None
    assert st['mvp']['name'] == 'Ali'
    assert st['mvp']['team'] == 'Falcons'
    e.close()


def test_football_goal_scorer_attribution():
    e = Engine()
    e.set_game('football')
    e.control({'command': 'default_goal'})
    e.control({'command': 'team', 'team': 'A', 'name': 'Falcons', 'players': ['Ali', 'Rashid']})
    e.control({'command': 'start'})
    frame1 = np.zeros((360, 640, 3), dtype=np.uint8)
    # Player Ali has ball outside goal zone (zone is around x=0.86, y=0.34, w=0.11, h=0.32 -> x=[550..620], y=[122..237])
    cv2.rectangle(frame1, (100, 100), (140, 180), (0, 0, 220), -1)
    cv2.circle(frame1, (130, 190), 8, (0, 255, 0), -1)
    e.process(frame1, now=100.0)
    e.process(frame1, now=100.1)

    # Ball shot into goal (x=580, y=180)
    frame_goal = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame_goal, (580, 180), 8, (0, 255, 0), -1)
    e.process(frame_goal, now=101.0)
    res = e.process(frame_goal, now=101.1)

    assert e.scores['A'] == 1
    assert e.scores['B'] == 0
    goal_event = [evt for evt in e.events if evt['kind'] == 'goal'][0]
    assert 'Ali' in goal_event['label']
    assert verify_report(e.report())[0]
    e.close()


def test_football_default_never_guesses_ball_or_goal():
    e = Engine()
    e.set_game('football')
    assert e.public_state()['goal_calibrated'] is False
    assert e.public_state()['football_ball_mode'] == 'yolo'
    e.close()


def test_wall_painting_rejected_as_ball():
    from vision import largest_round_marker
    # Rectangular flat green wall painting (e.g. 100x40 pixels)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (220, 135), (0, 255, 0), -1) # High aspect ratio, low circularity
    ball, _ = largest_round_marker(frame, (66, 160, 160))
    # Must be rejected because it is not round
    assert ball is None

    # Now draw an actual round ball (radius 10)
    cv2.circle(frame, (350, 200), 10, (0, 255, 0), -1)
    ball, _ = largest_round_marker(frame, (66, 160, 160))
    assert ball is not None
    assert abs(ball['x'] - (350 / 640)) < 0.05
    assert abs(ball['y'] - (200 / 360)) < 0.05


def test_hoodie_folds_merged_into_one_player():
    from vision import detect_team_players
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Simulate 3 separated folds/shadows on the same red hoodie
    cv2.rectangle(frame, (100, 120), (140, 190), (0, 0, 220), -1) # Chest right
    cv2.rectangle(frame, (145, 130), (180, 210), (0, 0, 220), -1) # Chest left
    cv2.rectangle(frame, (115, 200), (160, 260), (0, 0, 220), -1) # Lower torso
    players = detect_team_players(frame, (0, 210, 190), min_area=80)
    # The 3 overlapping/close fragments must be clustered into 1 single player box!
    assert len(players) == 1
    p = players[0]
    # Check that the merged box spans the entire person
    assert p['box'][0] <= (105 / 640)
    assert p['box'][1] <= (125 / 360)


def test_round_end_winner_and_cannot_restart():
    e=Engine()
    e.set_game('kancha')
    e.control({'command':'start'})
    e.process(board(x=10,y=100),now=100)
    e.process(board(x=300,y=170),now=101)
    e.process(board(x=300,y=170),now=102)
    e.control({'command':'finish'})
    assert e.public_state()['winner']=='Team A'
    assert e.public_state()['finished']
    import pytest
    with pytest.raises(ValueError):e.control({'command':'start'})
    assert verify_report(e.report())[0]
    e.close()


def test_football_team_possession_and_mvp():
    e = Engine()
    e.set_game('football')
    e.control({'command': 'team', 'team': 'A', 'name': 'Falcons', 'players': ['Ali', 'Rashid']})
    e.control({'command': 'team', 'team': 'B', 'name': 'Oryx', 'players': ['Zayed', 'Omar']})
    e.control({'command': 'start'})
    # Create a frame with Team A player (Red jersey: BGR = (0, 0, 220)) and neon green ball (BGR = (0, 255, 0))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Player A torso (Red)
    cv2.rectangle(frame, (100, 100), (140, 180), (0, 0, 220), -1)
    # Ball near player's feet (feet at ~ 120, 180)
    cv2.circle(frame, (130, 190), 8, (0, 255, 0), -1)
    # Process 5 consecutive frames
    for t in range(5):
        res = e.process(frame, now=100.0 + t * 0.1)
    st = e.public_state()
    assert st['team_possession']['A'] > 0.0
    assert st['team_possession']['B'] == 0.0
    assert st['possession_pct']['A'] == 100.0
    assert st['mvp'] is not None
    assert st['mvp']['name'] == 'Ali'
    assert st['mvp']['team'] == 'Falcons'
    e.close()


def test_football_goal_scorer_attribution():
    e = Engine()
    e.set_game('football')
    e.control({'command': 'default_goal'})
    e.control({'command': 'team', 'team': 'A', 'name': 'Falcons', 'players': ['Ali', 'Rashid']})
    e.control({'command': 'start'})
    frame1 = np.zeros((360, 640, 3), dtype=np.uint8)
    # Player Ali has ball outside goal zone (zone is around x=0.86, y=0.34, w=0.11, h=0.32 -> x=[550..620], y=[122..237])
    cv2.rectangle(frame1, (100, 100), (140, 180), (0, 0, 220), -1)
    cv2.circle(frame1, (130, 190), 8, (0, 255, 0), -1)
    e.process(frame1, now=100.0)
    e.process(frame1, now=100.1)

    # Ball shot into goal (x=580, y=180)
    frame_goal = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame_goal, (580, 180), 8, (0, 255, 0), -1)
    e.process(frame_goal, now=101.0)
    res = e.process(frame_goal, now=101.1)

    assert e.scores['A'] == 1
    assert e.scores['B'] == 0
    goal_event = [evt for evt in e.events if evt['kind'] == 'goal'][0]
    assert 'Ali' in goal_event['label']
    assert verify_report(e.report())[0]
    e.close()


def test_football_default_never_guesses_ball_or_goal():
    e = Engine()
    e.set_game('football')
    assert e.public_state()['goal_calibrated'] is False
    assert e.public_state()['football_ball_mode'] == 'yolo'
    e.close()


def test_wall_painting_rejected_as_ball():
    from vision import largest_round_marker
    # Rectangular flat green wall painting (e.g. 100x40 pixels)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (220, 135), (0, 255, 0), -1) # High aspect ratio, low circularity
    ball, _ = largest_round_marker(frame, (66, 160, 160))
    # Must be rejected because it is not round
    assert ball is None

    # Now draw an actual round ball (radius 10)
    cv2.circle(frame, (350, 200), 10, (0, 255, 0), -1)
    ball, _ = largest_round_marker(frame, (66, 160, 160))
    assert ball is not None
    assert abs(ball['x'] - (350 / 640)) < 0.05
    assert abs(ball['y'] - (200 / 360)) < 0.05


def test_hoodie_folds_merged_into_one_player():
    from vision import detect_team_players
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Simulate 3 separated folds/shadows on the same red hoodie
    cv2.rectangle(frame, (100, 120), (140, 190), (0, 0, 220), -1) # Chest right
    cv2.rectangle(frame, (145, 130), (180, 210), (0, 0, 220), -1) # Chest left
    cv2.rectangle(frame, (115, 200), (160, 260), (0, 0, 220), -1) # Lower torso
    players = detect_team_players(frame, (0, 210, 190), min_area=80)
    # The 3 overlapping/close fragments must be clustered into 1 single player box!
    assert len(players) == 1
    p = players[0]
    # Check that the merged box spans the entire person
    assert p['box'][0] <= (105 / 640)
    assert p['box'][1] <= (125 / 360)
