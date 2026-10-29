# FAMILY AI GAME DAY — Live AI Referee (Advanced Track)
*Abu Dhabi University Challenge · Grades 9–12 · 5 Games Track*

A full working AI referee and host application designed to fulfill the **Abu Dhabi University Family AI Game Day (Advanced Track)** requirements.

### Key Football Capabilities Implemented:
1. **Player & Ball Tracking**: Fast jersey color segmentation (Team A Red bibs, Team B Blue bibs, Neon green/orange ball) + optional YOLO26n tracking.
2. **Team Possession Percentage**: Real-time live possession bar showing exact seconds and percentage for Team A vs Team B.
3. **Player Possession & MVP Award**: Automatically tracks each player's ball control time; crowns the player with the most ball time as the **Most Valuable Player (MVP)**.
4. **Goal Detection & Scorer Attribution**: Debounced goal net detection that credits the goal to the attacking/possessing player and team.
5. **Triple Video Ingestion**:
   - 📷 **Live Webcam / Mobile Camera**
   - 📁 **Upload Match Video (.mp4 / .webm)**
   - ⚽ **Built-in Demo Match Clip (`demo_football.mp4`)** for instant 1-click testing without needing a camera.
6. **Audio Refereeing**: Recognizes referee speech cues ("Goal", "Start", "Pause", "Whistle").
7. **Tamper-Evident Report Verification**: SHA-256 hash chaining + server-side HMAC signatures, with an interactive drag-and-drop report verifier.

---

## Quickstart: Run on Windows

1. Double-click `start_windows.bat` in this folder.
2. Open your browser at **http://127.0.0.1:8000**.
3. Select **"⚽ Sample Match Clip"** and click **"▶ Play Sample Match"** followed by **"▶ Start Round"** to see the AI referee in action immediately!

To run tests:
```bat
.venv\Scripts\python.exe -m pytest tests
```

Open http://127.0.0.1:8000 in Chrome or Edge and select **Start camera**. Allow camera access. You should see a live feed, click-to-draw rectangle and real colour-marker detection. **Without AI packages/models, the football, Lagori and Kancha colour-vision demos work, but MediaPipe modes correctly refuse to fabricate scores.**

To add **trained AI pose/hand tracking** and optional football YOLO tracking, stop server (Ctrl+C):

```bat
.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The download script fetches the official pose and hand `.task` models directly from Google. Neither model file is bundled in this ZIP. Click **Enable YOLO player tracking** in Football to load the optional pretrained YOLO26n model; it may download weights on first use and needs a suitable internet connection. YOLO can be CPU-heavy; leave it off until the other three modes work.

If the `py -3.11` command doesn't work, first install Python 3.11 from python.org. For Mac/Linux replace `py -3.11` with `python3.11`, and `.venv\Scripts\python.exe` with `.venv/bin/python`.

## 2) Control guide

1. Select a game; fill Team A/B names, players and round duration; click **Save player setup**.
2. Start the webcam. Position it **fixed on a tripod**. Select **Draw zone**, then drag a rectangle over the scoring area.
3. Use neon-green game objects with a dark contrasting background, or click **Sample object color** and click an object in the live preview.
4. Choose the currently playing team, click **Start round**, and physically perform the game action. Events appear in the timeline, and points update automatically where supported.
5. Use **+1/−1 Correction** if an uncertain event was wrong, so the correction is logged; use **Approve/Reject catch** only for Sagala candidates.
6. **Download audit JSON** after a game, and click another game to archive its current results. Reports contain scores, old matches and event evidence.
7. Optional **Enable speech cues** uses browser speech recognition, which may use third-party cloud processing. Recognized words are logged as clues but do not earn points without video evidence.

Camera on a laptop via localhost works. On a **different phone on your LAN**, `http://192.168.x.x:8000` normally does **not** have the secure context needed by browser camera access; use a real HTTPS origin or suitable trusted local HTTPS setup. Do not expose this MVP over the internet without accounts, access controls and security hardening.

## 3) Five game-specific scoring algorithms

### Football (required)

**Physical setup:** small football goal, fixed side camera, bright green ball (or YOLO sports-ball object detection), red/blue team bibs. Draw the visible goal region. In a single-camera MVP, scoring is **a candidate goal** when the ball marker was previously outside the goal rectangle, enters it for **2 consecutive processed frames**, then waits until it exits before another point. Score goes to the team selected as currently attacking. This is a video event, **not proof of a legal goal-line crossing**.

**AI option:** Ultralytics YOLO26n tracks `person` and `sports ball` detections with ByteTrack. The nearest tracked player's foot-area to the ball is recorded as *estimated possession seconds*. Results include tracked player IDs and time. No automatic player-to-team jersey recognition, player name matching, or verified MVP determination is implemented yet. Improve this by training on match clips, team-colour calibration, tracking ball trajectory and the actual 3-D goal plane. Use a second camera if the rules require unequivocal goal verification.

### Lagori — India (competition adaptation)

**Physical setup:** seven green-painted blocks/discs stacked on dark mat, fixed camera. Draw a tight rectangle around the standing stack and select **Capture Lagori stack baseline**. The app computes the percentage of green pixels in the chosen region. If observed coverage falls below **55% of standing baseline for 3 processed frames**, it awards **1 provisional point** for the stack falling. Once coverage returns above **82% for 3 frames**, it awards **2 provisional rebuilding points**. Physical movement, lighting or occlusion can cause incorrect detections, so a person should review. This does not establish that an opposing ball caused the collapse.

### Kancha — India (competition adaptation)

**Physical setup:** overhead camera above the mat; use an enlarged bright-green marble and clearly visible goal target. Draw the target region. When the tracked marble is seen outside the target, then enters the target for **2 processed frames**, score **1 provisional point**. A stationary marble inside the pocket does not keep scoring; it must exit first. Use a very small camera field of view to keep each marble at least several pixels wide.

### Al Gaheef — UAE (competition adaptation)

**Physical setup:** 3×3 hopping grid with large marked squares; full body and feet visible. Draw the entire grid, which this prototype numbers **1 to 9 from top-left to bottom-right**. **Requires MediaPipe Pose**. It detects 33 body landmarks, checks if visible foot landmarks occupy the next expected cell for **3 processed frames**, awards **1 provisional point** per correctly visited cell and **3 extra** on route completion. 2-D vision alone **cannot establish that a foot touched the ground or that the other foot stayed lifted**. For a serious judge, add side-view camera or trained foot-contact classification, record the allowed hop pattern and validate a proper hop sequence.

### Al Sagala — UAE (competition adaptation)

**Physical setup:** overhead close-up on a dark mat and five large bright-green pebbles. **Requires MediaPipe Hands**. It detects up to two hands (21 hand landmarks each), estimates hand closing from finger positions, identifies green pebble blobs, and emits a **possible catch** when a recently raised hand closes near a visible pebble. **This is not reliable proof of a throw or a catch. No points are awarded automatically to these candidates.** A host must review and click **Approve catch** or **Reject catch**. For a fully automated submission, train or adapt a sequential toss–pebble–hand interaction classifier and compare it against annotated real throws/catches.

All scoring thresholds are prototypes and should be tuned with actual practice footage. They are not official rules for the traditional games. Preserve the authentic heritage context in your explanations and say explicitly that the event rules are your team's custom challenge variant.

## 4) Processing architecture

```
Browser camera + optional browser speech recognition
     │    640×360 JPEG at up to ~7 FPS (one in-flight frame)
     ▼
WebSocket /ws  ───► FastAPI app.py
     │                 │
     │                 ├──► OpenCV HSV colour segmentation
     │                 ├──► MediaPipe Tasks (pose/hand landmarks, optional)
     │                 └──► Ultralytics YOLO26n+ByteTrack (optional)
     │                              │
     │                       engine.py rule state machines
     │                              │
     ◄──── JSON with detections, scores, event evidence, confidence/review flag
     │
Overlay, scoreboard, match timer, event timeline, correction buttons, report
```

`vision.py`: detection/segmentation and optional trained models. `engine.py`: debounced, stateful game decisions, scores and hash-chained audit events. `app.py`: FastAPI HTTP+WebSocket service, report and report verification. `static/index.html`: browser interface.

All image processing happens on the student's computer running the Python server, not automatically on a remote cloud. A phone can act as the camera over a secured HTTPS connection to the computer.

## 5) Event evidence and verification

The Python judge writes each event to a hash chain (`previous_hash` + event content) and HMAC-signs each hash using the private server key. Exported JSON can be checked by the server using `POST /api/verify` or by this command while the same server is running:

```bat
.venv\Scripts\python.exe verify_report.py AI_Referee_xxx.json
```

To make signatures remain verifiable across server restarts, set a permanent secret environment variable **before** starting it, then keep the same key secret:

```bat
set JUDGE_SIGNING_KEY=replace-with-long-private-random-string
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

The key must stay on the trusted server. **A hash alone is not secure against someone recomputing hashes after editing a report**; the HMAC server-side signature adds meaningful integrity checking. This code is still not an independently audited evidence system.

## 6) Practical three-hour build/development plan

| Time | Priority | Acceptance condition |
|---|---|---|
| 0–30 min | Install, open app, camera, select game, test target colour | Green ball visibly tracked; no fabricated scores |
| 30–60 min | Calibrate Kancha + Lagori | Each scored action triggers exactly once; report contains real events |
| 60–100 min | Install ML models, calibrate Gaheef | Green overlay and detected foot landmarks move as person hops |
| 100–140 min | Test football, YOLO if available | Player IDs and plausible possession seconds; provisional goal events |
| 140–165 min | Test Sagala, approve uncertain catches | AI produces candidates on actual hand and pebble motion |
| 165–180 min | Record your demonstration with actual players and export report | Real footage, AI inference, scores and verified report all visible |

For a stronger competition entry, **validate and improve** football possession/MVP, the hop/contact rules, and Sagala catch judging rather than spending all the time on cosmetic features.

## 7) Testing and limitations

Run tests (pytest must be installed):

```bat
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest -q tests
```

The bundled tests use synthetic computer-vision scenes to verify score debouncing, baseline transitions, event integrity and match archiving. **They do not test actual webcam accuracy, YOLO/MediaPipe model downloads, different phones, voices or real-world game correctness.**

**Restrictions/risks:**
- Read the Ultralytics license before private or commercial deployment. Ultralytics code/models are generally AGPL-3.0 or Enterprise licensed.
- Get consent before filming participants (especially children); avoid posting personally identifying video without permission. Disable optional browser speech recognition unless needed; browser implementation may process audio remotely.
- No authentication, persistence, rate-limiting, server TLS, long-term storage or independent observer verification is implemented. Use only in a trusted local test environment.
- OpenCV HSV detection is not itself an ML model. It is a transparent supporting detector combined with MediaPipe/YOLO trained models where enabled; explain this distinction to judges.

## 8) Sources / technology documentation

- Competition attachment: *Competition Questions — Advanced Track (2026)*, pages 1–3.
- MediaPipe Pose Tasks, Python: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- MediaPipe Hand Tasks, Python: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- YOLO tracking: https://docs.ultralytics.com/modes/track
- Ultralytics license: https://www.ultralytics.com/license
- Camera security: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- MediaPipe official models are downloaded by `download_models.py`.

**Finishing a round:** Use **End round** to display the winning team or draw; the timer also ends the round automatically. **Pause** is temporary. **Reset** starts a new round. Switching games archives the previous game's event log and scores in the current session report.
