"""Download official MediaPipe pose and hand .task files (the models are not redistributed in the ZIP)."""
from pathlib import Path
from urllib.request import urlretrieve
ROOT=Path(__file__).parent/'models'
ROOT.mkdir(parents=True,exist_ok=True)
FILES={
    'pose_landmarker_lite.task':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'hand_landmarker.task':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
}
for name,url in FILES.items():
    target=ROOT/name
    if target.exists() and target.stat().st_size>100000:
        print('Already available:',target)
        continue
    print('Downloading:',name,'from',url)
    urlretrieve(url,target)
    if target.stat().st_size<100000: raise RuntimeError(f'Unexpectedly small model: {name}')
    print('Saved:',target)
print('Done. Restart the FastAPI server to load AI models.')
