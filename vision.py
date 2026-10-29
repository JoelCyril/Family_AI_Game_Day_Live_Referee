"""Computer-vision and OPTIONAL pretrained AI detectors.

Color segmentation helps with affordable, controlled physical game props.
MediaPipe Tasks is used for ML-derived pose/hand landmarks if model files exist.
Ultralytics YOLO is used for OPTIONAL person/ball tracking if installed.
These detectors are not themselves game referees; engine.py applies the rules.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent
MODELS = ROOT / 'models'
PRESETS = {
    'green': (66, 160, 160),    # Neon green ball / cones
    'orange': (14, 200, 200),   # Orange ball / bib
    'yellow': (30, 200, 200),   # Yellow ball / bib
    'red': (0, 210, 190),       # Red team bibs / shirts
    'blue': (115, 210, 190),    # Blue team bibs / shirts
    'purple': (140, 160, 140),
    'pink': (170, 160, 160),
    'white': (0, 20, 230),      # White shirts / ball
}


def hsv_from_bgr(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(map(int, cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0,0]))


def hsv_mask(frame: np.ndarray, hsv: tuple[int, int, int], tolerance: int = 15) -> np.ndarray:
    fhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in hsv)
    # White / low-saturation objects
    if s < 45:
        mask = cv2.inRange(fhsv, (0, 0, max(160, v - 60)), (179, min(70, s + 40), 255))
    else:
        s_min, v_min = max(45, s - 90), max(40, v - 110)
        lo, hi = (h - tolerance) % 180, (h + tolerance) % 180
        if lo <= hi:
            mask = cv2.inRange(fhsv, (lo, s_min, v_min), (hi, 255, 255))
        else:
            mask = cv2.inRange(fhsv, (0, s_min, v_min), (hi, 255, 255)) | cv2.inRange(fhsv, (lo, s_min, v_min), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask


def box_iou(a: list[float], b: list[float]) -> float:
    """Computes Intersection-over-Union between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union > 0 else 0.0


def merge_close_boxes(boxes: list[list[int]], dist_threshold: int = 35) -> list[list[int]]:
    """Merges close or overlapping bounding boxes [x, y, w, h] belonging to the same body/clothing."""
    if not boxes:
        return []
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if not used[j]:
                    x2, y2, w2, h2 = boxes[j]
                    # Check if boxes overlap or are within distance threshold
                    if not (x1 + w1 + dist_threshold < x2 or x2 + w2 + dist_threshold < x1 or
                            y1 + h1 + dist_threshold < y2 or y2 + h2 + dist_threshold < y1):
                        # Merge boxes
                        nx1 = min(x1, x2)
                        ny1 = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx1
                        nh = max(y1 + h1, y2 + h2) - ny1
                        x1, y1, w1, h1 = nx1, ny1, nw, nh
                        used[j] = True
                        changed = True
        merged.append([x1, y1, w1, h1])
    return merged


def detect_team_players(frame: np.ndarray, team_hsv: tuple[int, int, int], min_area: int = 120, max_players: int = 5) -> list[dict]:
    """Detects players of a given team by jersey/bib color with body-clustering to prevent multi-box fragmentation."""
    mask = hsv_mask(frame, team_hsv, tolerance=15)
    # Morphological CLOSE with large kernel bridges zippers, clothing folds, and torso shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_f, w_f = frame.shape[:2]
    raw_boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            raw_boxes.append(list(cv2.boundingRect(c)))
    
    # Merge close boxes from the same person
    merged_boxes = merge_close_boxes(raw_boxes, dist_threshold=int(w_f * 0.05))
    
    # Sort by area descending
    merged_boxes = sorted(merged_boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_players]
    
    players = []
    for box in merged_boxes:
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        # Feet are approximated at the bottom center of the bounding box
        players.append({
            'x': round(cx / w_f, 4),
            'y': round(cy / h_f, 4),
            'foot': [round(cx / w_f, 4), round(min(1.0, (y + h) / h_f), 4)],
            'box': [round(x / w_f, 4), round(y / h_f, 4), round(w / w_f, 4), round(h / h_f, 4)],
            'area': round(float(w * h), 1)
        })
    return players


def largest_round_marker(frame: np.ndarray, hsv: tuple[int, int, int]):
    """Strict circular ball prop detector. Rejects flat wall paintings, furniture, and non-round shapes."""
    mask = hsv_mask(frame, hsv, tolerance=14)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # Size sanity checks (avoid full screens or tiny specks)
        if area < 30 or area > 0.012 * width * height:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        # Circularity ratio: 4 * pi * area / (perimeter^2) -> 1.0 is a perfect circle
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.60:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Aspect ratio: ball must be reasonably square/circular
        if w < 6 or h < 6 or not (0.68 <= (w / h) <= 1.47):
            continue
        center, radius = cv2.minEnclosingCircle(cnt)
        if radius < 3 or radius > min(height, width) * 0.09:
            continue
        candidates.append((area, center, (x, y, w, h), circularity))
    
    if not candidates:
        return None, mask
    # Choose candidate with highest roundness & plausible size
    _, (cx, cy), (x, y, w, h), circ = max(candidates, key=lambda p: p[0])
    return {
        'x': round(cx / width, 4),
        'y': round(cy / height, 4),
        'box': [round(x / width, 4), round(y / height, 4), round(w / width, 4), round(h / height, 4)],
        'circularity': round(float(circ), 2)
    }, mask


def classify_person_team(frame: np.ndarray, box: list[float], hsv_a: tuple[int, int, int], hsv_b: tuple[int, int, int]) -> str | None:
    """Classifies a detected person box into Team A or Team B based on torso color."""
    h_f, w_f = frame.shape[:2]
    x, y, w, h = [int(box[0]*w_f), int(box[1]*h_f), int(box[2]*w_f), int(box[3]*h_f)]
    torso_y0 = max(0, min(h_f - 1, int(y + 0.12 * h)))
    torso_y1 = max(0, min(h_f, int(y + 0.55 * h)))
    torso_x0 = max(0, min(w_f - 1, int(x + 0.12 * w)))
    torso_x1 = max(0, min(w_f, int(x + 0.88 * w)))
    if torso_y1 <= torso_y0 or torso_x1 <= torso_x0:
        return None
    torso = frame[torso_y0:torso_y1, torso_x0:torso_x1]
    cnt_a = cv2.countNonZero(hsv_mask(torso, hsv_a, tolerance=16))
    cnt_b = cv2.countNonZero(hsv_mask(torso, hsv_b, tolerance=16))
    if cnt_a > cnt_b and cnt_a > 12:
        return 'A'
    elif cnt_b > cnt_a and cnt_b > 12:
        return 'B'
    return None


def largest_marker(frame: np.ndarray, hsv: tuple[int,int,int], min_area: int = 15):
    mask = hsv_mask(frame, hsv)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    m = cv2.moments(c)
    if not m['m00']:
        return None, mask
    x, y, w, h = cv2.boundingRect(c)
    cx,cy = m['m10']/m['m00'], m['m01']/m['m00']
    return {'x': round(cx/frame.shape[1],4),'y':round(cy/frame.shape[0],4),
            'box':[round(x/frame.shape[1],4),round(y/frame.shape[0],4),round(w/frame.shape[1],4),round(h/frame.shape[0],4)],
            'area':round(float(cv2.contourArea(c)),1)}, mask


def coverage(frame: np.ndarray, hsv: tuple[int,int,int], zone: list[float]):
    h,w = frame.shape[:2]
    x,y,z_w,z_h = zone
    x0,x1 = max(0,int(x*w)),min(w,int((x+z_w)*w))
    y0,y1 = max(0,int(y*h)),min(h,int((y+z_h)*h))
    if x1<=x0 or y1<=y0: return 0.0
    mask = hsv_mask(frame[y0:y1,x0:x1], hsv)
    return float(cv2.countNonZero(mask))/float(mask.size)


def in_zone(point: tuple[float,float] | None, zone: list[float]) -> bool:
    if point is None: return False
    x,y,w,h = zone
    return x<=point[0]<=x+w and y<=point[1]<=y+h


def detect_blobs(frame: np.ndarray, hsv: tuple[int,int,int], zone: list[float] | None = None, min_area: int = 8) -> list[dict]:
    mask = hsv_mask(frame,hsv)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cnts:
        if cv2.contourArea(c)<min_area: continue
        m=cv2.moments(c)
        if not m['m00']: continue
        pt=(m['m10']/m['m00']/frame.shape[1],m['m01']/m['m00']/frame.shape[0])
        if zone is None or in_zone(pt,zone):out.append({'x':round(pt[0],4),'y':round(pt[1],4),'area':round(float(cv2.contourArea(c)),1)})
    return sorted(out,key=lambda o:o['area'],reverse=True)


class PoseHands:
    """Uses MediaPipe TASKS, video mode. Lazy initialization once per session."""
    def __init__(self):
        self.mp = None
        self.pose = None
        self.hands = None
        self.pose_reason = 'Pose model unavailable. Install mediapipe and download the .task model.'
        self.hands_reason = 'Hand model unavailable. Install mediapipe and download the .task model.'
        try:
            import mediapipe as mp
            self.mp=mp
        except ImportError:
            return
        BaseOptions=mp.tasks.BaseOptions
        R=mp.tasks.vision.RunningMode
        pp=MODELS/'pose_landmarker_lite.task'
        hh=MODELS/'hand_landmarker.task'
        if pp.exists():
            try:
                opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(pp)), running_mode=R.VIDEO,num_poses=1)
                self.pose=mp.tasks.vision.PoseLandmarker.create_from_options(opts)
                self.pose_reason='MediaPipe Pose Landmarker loaded'
            except Exception as ex:
                self.pose_reason=f'Pose model failed: {ex}'
        if hh.exists():
            try:
                opts=mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(hh)), running_mode=R.VIDEO,num_hands=2)
                self.hands=mp.tasks.vision.HandLandmarker.create_from_options(opts)
                self.hands_reason='MediaPipe Hand Landmarker loaded'
            except Exception as ex:
                self.hands_reason=f'Hand model failed: {ex}'

    def landmarks(self, frame: np.ndarray, timestamp_ms: int, task: str) -> list[list[dict]]:
        model = self.pose if task=='pose' else self.hands
        if model is None: return []
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
        result=model.detect_for_video(img, timestamp_ms)
        groups=result.pose_landmarks if task=='pose' else result.hand_landmarks
        return [[{'x':float(p.x),'y':float(p.y),'visibility':float(getattr(p,'visibility',1.0) or 0.0)} for p in group] for group in groups]

    def close(self):
        if self.pose: self.pose.close()
        if self.hands: self.hands.close()


class OptionalYolo:
    def __init__(self):
        self.model=None
        self.reason='YOLO disabled (optional; install ultralytics and enable in app).'
    def enable(self):
        try:
            from ultralytics import YOLO
            self.model=YOLO('yolo26n.pt')  # Downloads weights on first use.
            self.reason='YOLO26n tracking enabled'
        except Exception as ex:
            self.reason=f'YOLO unavailable: {ex}'
        return self.model is not None
    def analyze(self, frame:np.ndarray):
        if self.model is None:return [],None
        try:
            result=self.model.track(frame, persist=True, classes=[0,32], tracker='bytetrack.yaml', conf=.30, verbose=False)[0]
            people=[]
            ball=None
            if result.boxes is None: return people,ball
            boxes=result.boxes
            ids=boxes.id.int().cpu().tolist() if boxes.id is not None else [-1]*len(boxes)
            for coords, cls, conf, pid in zip(boxes.xyxy.cpu().tolist(),boxes.cls.int().cpu().tolist(),boxes.conf.cpu().tolist(),ids):
                x0,y0,x1,y1=[int(c) for c in coords]
                if cls==0 and conf>=.45:
                    people.append({'id':int(pid),'box':[x0/frame.shape[1],y0/frame.shape[0],(x1-x0)/frame.shape[1],(y1-y0)/frame.shape[0]],'confidence':round(conf,2)})
                if cls==32 and conf>=.35 and (ball is None or conf>ball['confidence']):
                    ball={'x':(x0+x1)/2/frame.shape[1],'y':(y0+y1)/2/frame.shape[0], 'confidence':round(conf,2)}
            # Suppress overlapping duplicate person boxes using IoU NMS
            filtered = []
            for person in sorted(people, key=lambda p: -p['confidence']):
                if all(box_iou(person['box'], q['box']) < 0.50 for q in filtered):
                    filtered.append(person)
            return filtered, ball
        except Exception as ex:
            self.reason=f'YOLO track error: {ex}'
            return [],None
