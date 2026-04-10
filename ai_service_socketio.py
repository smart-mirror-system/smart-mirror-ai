import os
import time
import json
import cv2
import jwt
import socketio

from dotenv import load_dotenv
from exercise_counters import ExerciseCounter
from core.rtmpose_processor import RTMPoseProcessor

load_dotenv(override=True)
print("[DEBUG] RUN_MODE env =", os.getenv("RUN_MODE"))
# =========================
# Config (env first, then defaults)
# =========================
RUN_MODE = os.getenv("RUN_MODE", "socketio").strip().lower()  # socketio | standalone
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000").strip()
TOKEN = os.getenv("DEVICE_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("Missing DEVICE_TOKEN (device JWT).")

EXERCISE_TYPE = os.getenv("EXERCISE_TYPE", "pushup").strip()  # used as default / standalone
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SEND_EVERY_MS = int(os.getenv("SEND_EVERY_MS", "250"))  # throttle updates
MODEL_MODE = os.getenv("MODEL_MODE", "lightweight").strip()  # lightweight / balanced / performance
SHOW_CAMERA = os.getenv("SHOW_CAMERA", "0") == "1"
# Optional debug export (legacy JSON streaming / debugging)
EXPORT_JSON = os.getenv("EXPORT_JSON", "0") == "1"
EXPORT_JSON_PATH = os.getenv("EXPORT_JSON_PATH", "live_stream_data.json").strip()

if RUN_MODE not in ("socketio", "standalone"):
    raise SystemExit("RUN_MODE must be 'socketio' or 'standalone'")

if RUN_MODE == "socketio" and not TOKEN:
    raise SystemExit("Missing DEVICE_TOKEN (recommended) or AI_JWT (dev fallback) for socketio mode.")

# Decode token payload WITHOUT verifying signature (for logs only).
# Server will do real verification.
ID_HINT = "unknown"
if TOKEN:
    try:
        p = jwt.decode(TOKEN, options={"verify_signature": False})
        ID_HINT = str(
            p.get("deviceId")
            or p.get("userId")
            or p.get("id")
            or p.get("_id")
            or "unknown"
        )
    except Exception:
        pass

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

# Runtime state controlled by backend commands
current_user_id = None
current_exercise = EXERCISE_TYPE
running = False
cap = None
destroy_window_requested = False


@sio.event
def connect():
    print(f"[AI] Connected to backend: {BACKEND_URL} (id_hint={ID_HINT})")


@sio.event
def connect_error(data):
    print("[AI] connect_error:", data)


@sio.event
def disconnect():
    print("[AI] Disconnected")


@sio.on("workout:start")
def on_start(data):
    global current_user_id, current_exercise, running
    current_user_id = str(data.get("userId"))
    current_exercise = str(data.get("exerciseType") or EXERCISE_TYPE)
    # Reset will happen in main loop when starting
    running = True
    print(f"[AI] workout:start user={current_user_id} ex={current_exercise}")

@sio.on("workout:stop")
def on_stop(data):
    global running, destroy_window_requested
    running = False
    if SHOW_CAMERA:
        destroy_window_requested = True
    print(f"[AI] workout:stop user={data.get('userId')}")


def safe_form_score(angle):
    """
    Placeholder form score (0-100).
    Replace later with real quality metrics.
    """
    if angle is None:
        return 0
    return max(10, min(100, int(100 - abs(angle - 120) * 0.5)))


def export_debug_json(payload: dict):
    """
    Optional JSON export for debugging only.
    Controlled by EXPORT_JSON=1.
    """
    if not EXPORT_JSON:
        return
    try:
        with open(EXPORT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[AI] export_debug_json failed:", e)


def reset_counter(counter: ExerciseCounter):
    """
    Reset counter safely without depending on specific class implementation.
    """
    if hasattr(counter, "reset_counter") and callable(getattr(counter, "reset_counter")):
        try:
            counter.reset_counter()
            return
        except Exception:
            pass
    try:
        counter.counter = 0
        counter.stage = None
        counter.angle_history.clear()
        counter.leg_stages = {'left': None, 'right': None}
    except Exception:
        pass

def handle_destroy_window():
    global destroy_window_requested
    if destroy_window_requested and SHOW_CAMERA:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        destroy_window_requested = False

def main():
    global running

    # 1) Init counter + pose processor
    counter = ExerciseCounter()
    processor = RTMPoseProcessor(exercise_counter=counter, mode=MODEL_MODE)

    # 2) Connect to backend (socketio mode only)
    if RUN_MODE == "socketio":
        try:
            sio.connect(
                BACKEND_URL,
                transports=["websocket"],
                auth={"token": TOKEN},
            )
        except Exception as e:
            print("[AI] Could not connect to backend:", e)
            raise SystemExit("Backend is not running or BACKEND_URL is wrong.")
    else:
        print("[AI] Standalone mode (no backend).")
        running = True  # start immediately in standalone

    # 3) Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Check CAMERA_INDEX or camera permissions.")

    last_sent_ms = 0
    last_reps_sent = -1
    last_printed_reps = -1

    print(f"[AI] Ready. mode={RUN_MODE}, default_ex={EXERCISE_TYPE}, camera={CAMERA_INDEX}, show_camera={SHOW_CAMERA}")

    try:
        while True:
            if not running:
                handle_destroy_window()  # main thread يقفل الـ window لو في طلب
                time.sleep(0.05)
                continue

            # When a workout starts, reset counter once at the beginning
            # (simple way: detect transition by using last_reps_sent == -1)
            if last_reps_sent == -1:
                reset_counter(counter)

            ok, frame = cap.read()
            if not ok:
                continue

            # Optional resize for speed
            frame = cv2.resize(frame, (320, 240))

            exercise = current_exercise or EXERCISE_TYPE

            # Process pose
            # Returns: (annotated_frame?, angle, stage?, keypoints)
            _img, angle, _unused, _keypoints = processor.process_frame(frame, exercise)

            reps = int(getattr(counter, "counter", 0))
            stage = getattr(counter, "stage", None) or "unknown"

            # ===== optional debug window =====
            if SHOW_CAMERA:
                cv2.putText(frame, f"EX: {exercise}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reps: {reps}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {stage}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("AI Debug Camera", frame)
                # Press 'q' to stop current workout (does not exit program)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False
                    # reset for next session
                    last_reps_sent = -1
                    continue

            now_ms = int(time.time() * 1000)

            should_send = (reps != last_reps_sent) or (now_ms - last_sent_ms >= SEND_EVERY_MS)

            if running and current_user_id and should_send:
                last_sent_ms = now_ms
                last_reps_sent = reps

                payload = {
                    "userId": current_user_id,
                    "exerciseType": current_exercise,
                    "reps": reps,
                    "stage": stage,
                    "angle": float(angle) if angle is not None else 0,
                    "formScore": safe_form_score(angle),
                    "mistakes": [],
                    "ts": now_ms,
                }

                # Optional local debug export
                export_debug_json(payload)

                if RUN_MODE == "socketio" and sio.connected:
                    try:
                        sio.emit("ai:progress", payload)
                    except Exception as e:
                        print("[AI] emit failed:", e)
                else:
                    # standalone: print on rep change
                    if reps != last_printed_reps:
                        last_printed_reps = reps
                        print(f"[AI] {exercise}: reps={reps}, stage={stage}, score={payload['formScore']}")

            # If workout stopped by backend, reset session markers
            if not running:
                last_reps_sent = -1
                handle_destroy_window()

    except KeyboardInterrupt:
        print("\n[AI] Stopping...")
    finally:
        if SHOW_CAMERA:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            cap.release()
        except Exception:
            pass

        if RUN_MODE == "socketio":
            try:
                sio.disconnect()
            except Exception:
                pass

if __name__ == "__main__":
    main()