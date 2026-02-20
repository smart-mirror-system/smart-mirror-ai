# ai_service_socketio.py
# Headless AI service: camera -> pose -> count -> Socket.IO backend

import os
import time
import cv2
import jwt  # PyJWT
import socketio

from exercise_counters import ExerciseCounter
from core.rtmpose_processor import RTMPoseProcessor
from dotenv import load_dotenv
load_dotenv()

# =========================
# Config (env first, then defaults)
# =========================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
AI_JWT = os.getenv("AI_JWT", "").strip()
EXERCISE_TYPE = os.getenv("EXERCISE_TYPE", "pushup").strip()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SEND_EVERY_MS = int(os.getenv("SEND_EVERY_MS", "250"))  # throttle updates
MODEL_MODE = os.getenv("MODEL_MODE", "lightweight") 
SHOW_CAMERA = os.getenv("SHOW_CAMERA", "0") == "1"

if not AI_JWT:
    raise SystemExit("Missing AI_JWT env var (use the token from /api/auth/login).")

# Decode userId from JWT payload WITHOUT verifying signature (server will verify anyway).
# This is only to send the required userId for room:join / ai:progress due to current backend contract.
payload = jwt.decode(AI_JWT, options={"verify_signature": False})
USER_ID = str(payload.get("userId") or payload.get("id") or payload.get("_id") or "")
if not USER_ID:
    raise SystemExit("JWT payload does not contain userId/id/_id.")

# =========================
# Socket.IO client
# =========================
sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=999999,
    reconnection_delay=1,
    logger=False,
    engineio_logger=False,
)

@sio.event
def connect():
    print("[AI] Connected to backend:", BACKEND_URL)
    # Current backend expects room:join with userId (it re-checks against token).
    sio.emit("room:join", {"userId": USER_ID})
    print("[AI] Joined room for user:", USER_ID)

@sio.event
def connect_error(data):
    print("[AI] connect_error:", data)

@sio.event
def disconnect():
    print("[AI] Disconnected")

def safe_form_score(angle):
    """
    Simple placeholder form score (0-100). Real scoring can be added later.
    """
    if angle is None:
        return 0
    # A naive heuristic: closer to mid-range is better (just for demo).
    # You can replace with real CV quality metrics later.
    return max(10, min(100, int(100 - abs(angle - 120) * 0.5)))

def main():
    # 1) Init counter + pose processor (same components used in main.py microservice)
    counter = ExerciseCounter()
    processor = RTMPoseProcessor(exercise_counter=counter, mode=MODEL_MODE)

    # 2) Connect to backend with auth token (same as your mock-ai.js logic)
    sio.connect(
        BACKEND_URL,
        transports=["websocket"],
        auth={"token": AI_JWT},
    )

    # 3) Open camera (headless - no cv2.imshow
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise SystemExit("Could not open camera. Check CAMERA_INDEX or camera permissions.")
    last_sent_ms = 0
    last_reps = -1

    print(f"[AI] Running headless. exercise={EXERCISE_TYPE}, camera={CAMERA_INDEX}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
        
            # Optional: resize for speed
            frame = cv2.resize(frame, (320, 240))

            # process_frame updates
            _, angle, _, keypoints = processor.process_frame(frame, EXERCISE_TYPE)

            reps = int(counter.counter)
            stage = counter.stage if counter.stage is not None else "unknown"

            # ===== optional debug window =====
            if SHOW_CAMERA:
                # Small overlay for debugging
                cv2.putText(frame, f"EX: {EXERCISE_TYPE}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reps: {reps}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {stage}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("AI Debug Camera", frame)
                
                # Press 'q' to stop debug window (optional)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now_ms = int(time.time() * 1000)

            # Send when reps changed OR periodically (throttled)
            should_send = (reps != last_reps) or (now_ms - last_sent_ms >= SEND_EVERY_MS)
            if should_send:
                last_sent_ms = now_ms
                last_reps = reps

                payload = {
                    "userId": USER_ID,
                    "exerciseType": EXERCISE_TYPE,
                    "reps": reps,
                    "stage": stage,
                    "angle": float(angle) if angle is not None else 0,
                    "formScore": safe_form_score(angle),
                    "mistakes": [],  # TODO: add real mistakes later
                    "ts": now_ms,
                    # Optional: send skeleton coords (can be heavy)
                    # "skeleton": keypoints.tolist() if keypoints is not None else [],
                }

                if sio.connected:
                    try:
                        sio.emit("ai:progress", payload)
                    except Exception as e:
                        print("[AI] emit failed:", e)
                else:
                    print("[AI] Not connected, skipping emit")

    except KeyboardInterrupt:
        print("\n[AI] Stopping...")
    finally:
        if SHOW_CAMERA:
            cv2.destroyAllWindows()
        cap.release()
        try:
            sio.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
