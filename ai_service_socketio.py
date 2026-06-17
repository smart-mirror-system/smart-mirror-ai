import os
import time
import json
import cv2
from numpy import angle
import jwt
import socketio
import math
import pyautogui

from dotenv import load_dotenv
from exercise_counters import ExerciseCounter
from core.rtmpose_processor import RTMPoseProcessor
import mediapipe as mp  # Compatible with stable release 0.10.9

load_dotenv(override=True)
print("[DEBUG] RUN_MODE env =", os.getenv("RUN_MODE"))

# =========================
# Config
# =========================
RUN_MODE = os.getenv("RUN_MODE", "socketio").strip().lower()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000").strip()
TOKEN = os.getenv("DEVICE_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("Missing DEVICE_TOKEN (device JWT).")

EXERCISE_TYPE = os.getenv("EXERCISE_TYPE", "pushup").strip()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SEND_EVERY_MS = int(os.getenv("SEND_EVERY_MS", "250"))
MODEL_MODE = os.getenv("MODEL_MODE", "lightweight").strip()
SHOW_CAMERA = os.getenv("SHOW_CAMERA", "1") == "1"  # Keep it 1 by default for a clear visual test
EXPORT_JSON = os.getenv("EXPORT_JSON", "0") == "1"
EXPORT_JSON_PATH = os.getenv("EXPORT_JSON_PATH", "live_stream_data.json").strip()

if RUN_MODE not in ("socketio", "standalone"):
    raise SystemExit("RUN_MODE must be 'socketio' or 'standalone'")

# Decode the token for debugging
ID_HINT = "unknown"
if TOKEN:
    try:
        p = jwt.decode(TOKEN, options={"verify_signature": False})
        ID_HINT = str(p.get("deviceId") or p.get("userId") or p.get("id") or p.get("_id") or "unknown")
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

# Runtime state
current_user_id = "standalone_user"  # Default value so the code runs even without a backend in standalone mode
current_exercise = EXERCISE_TYPE
cap = None
destroy_window_requested = False

# 🚨 New magic variable to manage gestures and automatic navigation:
# 0 = Navigation Mode (Hand Mouse code active and workouts stopped)
# 1 = Workout Mode (mouse disabled, RTMPose code counts, waiting for stop signal)
CURRENT_MODE = 0 

# Stop gesture hold time calculations
stop_signal_start_time = None
frame_reduction = 80
screen_width, screen_height = pyautogui.size()

# Exercise processor (RTMPose + counter) initialized later in main() after the Socket.IO connection is established, so we can send the mapping immediately on connect
processor = None

@sio.event
def connect():
    print(f"[AI] Connected to backend: {BACKEND_URL} (id_hint={ID_HINT})")

@sio.event
def connect_error(data):
    print("[AI] connect_error:", data)

@sio.event
def disconnect():
    print("[AI] Disconnected")

def connect():
    global processor # We need to access the processor instance to send the mapping right after connecting
    print("Connected to server!")
    if processor:
        mapping = processor.get_keypoint_mapping()
        sio.emit("ai:mapping", mapping)
        print("Sent keypoint mapping to server.")

# Backend can also change the mode if an admin uses the control panel without the screen
@sio.on("workout:start")
def on_start(data):
    global current_user_id, current_exercise, CURRENT_MODE
    current_user_id = str(data.get("userId"))
    current_exercise = str(data.get("exerciseType") or EXERCISE_TYPE)
    CURRENT_MODE = 1
    print(f"[AI] Backend triggered workout:start user={current_user_id} ex={current_exercise}")

@sio.on("workout:stop")
def on_stop(data):
    global CURRENT_MODE, destroy_window_requested
    CURRENT_MODE = 0
    if SHOW_CAMERA:
        destroy_window_requested = True
    print(f"[AI] Backend triggered workout:stop")


def safe_form_score(angle):
    if angle is None or math.isnan(angle):
        return 0
    return max(10, min(100, int(100 - abs(angle - 120) * 0.5)))

def reset_counter(counter: ExerciseCounter):
    if hasattr(counter, "reset_counter") and callable(getattr(counter, "reset_counter")):
        try: counter.reset_counter(); return
        except Exception: pass
    try:
        counter.counter = 0
        counter.stage = None
        counter.angle_history.clear()
        counter.leg_stages = {'left': None, 'right': None}
    except Exception: pass

def main():
    global CURRENT_MODE, current_user_id, current_exercise, stop_signal_start_time, processor
    # 🚨 Disable the fail-safe completely so the mouse doesn't close the program if it goes to a screen corner
    pyautogui.FAILSAFE = False
    # 1) Load models (RTMPose + shared MediaPipe Hands)
    counter = ExerciseCounter()
    processor = RTMPoseProcessor(exercise_counter=counter, mode=MODEL_MODE)
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_model = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # 2) Connect to the backend
    if RUN_MODE == "socketio":
        try:
            sio.connect(BACKEND_URL, transports=["websocket"], auth={"token": TOKEN})
        except Exception as e:
            print("[AI] Could not connect to backend:", e)
            raise SystemExit("Backend is not running or BACKEND_URL is wrong.")
    else:
        print("[AI] Standalone mode (no backend).")

    # 3) Open the camera immediately at startup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Check CAMERA_INDEX.")

    last_sent_ms = 0
    last_reps_sent = -1
    last_printed_reps = -1

    print(f"[AI] System Running. Initial Mode: Navigation (Hand Mouse active)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue

            # Flip the frame horizontally so the user feels it is a real mirror
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ========================================================
            # Mode [0]: browsing mode with Hand Mouse active
            # ========================================================
            if CURRENT_MODE == 0:
                hand_results = hands_model.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if SHOW_CAMERA:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # 1. Fetch finger points (stable MediaPipe version)
                        thumb_finger = hand_landmarks.landmark[4]   # thumb
                        index_finger = hand_landmarks.landmark[8]   # index (for move/click)
                        middle_finger = hand_landmarks.landmark[12] # middle (for scroll)
                        
                        # 2. Compute actual mouse movement (based on the index finger)
                        x_mouse = int((index_finger.x * w - frame_reduction) * screen_width / (w - 2 * frame_reduction))
                        y_mouse = int((index_finger.y * h - frame_reduction) * screen_height / (h - 2 * frame_reduction))
                        
                        # Protect the deadly corner boundaries
                        x_mouse = max(10, min(screen_width - 10, x_mouse))
                        y_mouse = max(10, min(screen_height - 10, y_mouse))
                        
                        # 3. Calculate distances between fingers in pixels
                        # Click distance: thumb + index
                        click_dist = math.hypot(int(thumb_finger.x*w) - int(index_finger.x*w), int(thumb_finger.y*h) - int(index_finger.y*h))
                        
                        # Scroll distance: thumb + middle
                        scroll_dist = math.hypot(int(thumb_finger.x*w) - int(middle_finger.x*w), int(thumb_finger.y*h) - int(middle_finger.y*h))
                        
                        # 4. Execute actions based on gestures
                        
                        # 🚨 [First action]: Scroll (thumb + middle)
                        if scroll_dist < 30:
                            # To decide whether to scroll up or down, compare finger position with the vertical center of the screen (h / 2)
                            # Or simpler: if the finger is in the upper half, scroll up; if in the lower half, scroll down
                            if int(middle_finger.y * h) < (h / 2):
                                pyautogui.scroll(150) # Scroll up
                                print("[AI Gesture] Scrolling UP ⬆️")
                            else:
                                pyautogui.scroll(-150) # Scroll down
                                print("[AI Gesture] Scrolling DOWN ⬇️")
                                
                            pyautogui.sleep(0.1) # Small delay so scrolling is not too fast or jumpy
                        
                        # 👆 [Second action]: Normal click (thumb + index)
                        elif click_dist < 30:
                            pyautogui.click()
                            print("[AI Gesture] Mouse Click Executed!")
                            pyautogui.sleep(0.4)
                            
                        # 🖱️ [Normal state]: move the cursor when there is no click or scroll
                        else:
                            pyautogui.moveTo(x_mouse, y_mouse, _pause=False)

                if SHOW_CAMERA:
                    cv2.putText(frame, "Mode: Navigation (Mouse + Scroll)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imshow("AI Debug Camera", frame)
    
            # ========================================================
            # Mode [1]: workout mode with RTMPose active (mouse fully disabled)
            # ========================================================
            elif CURRENT_MODE == 1:
                if last_reps_sent == -1:
                    reset_counter(counter)
                    # Protection: Record workout start time to prevent immediate locking and freezing during the first 5 seconds.
                    workout_start_telemetry = time.time()

                _img, angle, _unused, keypoints = processor.process_frame(frame, current_exercise)

                reps = int(getattr(counter, "counter", 0))
                stage = getattr(counter, "stage", None) or "unknown"

                # Period of allowance 5 seconds: in the first exercise it is impossible to lock so that you can catch up, return and prepare the camera
                time_elapsed_since_start = time.time() - workout_start_telemetry
                
                if time_elapsed_since_start > 5.0:
                    try:
                        if keypoints is not None and len(keypoints) > 10:
                            # Fetch keypoint coordinates using RTMPose COCO format
                            nose_y = keypoints[0][1]
                            left_shoulder_x = keypoints[5][0]
                            right_shoulder_x = keypoints[6][0]
                            left_wrist_y = keypoints[9][1]
                            right_wrist_y = keypoints[10][1]
                            
                            # Calculate shoulder width to determine proximity to the camera
                            shoulder_width = abs(left_shoulder_x - right_shoulder_x)
                            
                            # Strict conditions: proximity to the camera + hand raised above nose level 🖐️
                            is_close = shoulder_width > 150  # This value should be adjusted based on camera size and room dimensions
                            hand_raised = (left_wrist_y < nose_y) or (right_wrist_y < nose_y)
                            
                            if is_close and hand_raised:
                                if stop_signal_start_time is None:
                                    stop_signal_start_time = time.time()
                                else:
                                    # ⏱️ Full Safety Check: Must stay close with hands raised for 3 full seconds
                                    if time.time() - stop_signal_start_time > 3.0:
                                        print("🚨 [AI Smart Stop] Cancel Triggered Successfully after 3 seconds of stability!")
                                        if RUN_MODE == "socketio" and sio.connected:
                                            sio.emit("workout:cancel", {"userId": current_user_id})
                                            
                                        CURRENT_MODE = 0
                                        last_reps_sent = -1
                                        stop_signal_start_time = None
                                        pyautogui.sleep(1.5)
                                        continue
                            else:
                                # If hands are lowered or user moves away before 3 seconds, reset the timer immediately
                                stop_signal_start_time = None
                        else:
                            stop_signal_start_time = None
                    except Exception as e:
                        print("[AI Debug] Stop detection error:", e)
                        stop_signal_start_time = None
                else:
                    # While we are in the first 5 seconds, the timer is reset and cannot be triggered
                    stop_signal_start_time = None
                
                # Send reps data to the backend via socket
                now_ms = int(time.time() * 1000)
                should_send = (reps != last_reps_sent) or (now_ms - last_sent_ms >= SEND_EVERY_MS)

                if current_user_id and should_send:
                    last_sent_ms = now_ms
                    last_reps_sent = reps

                    payload = {
                        "userId": current_user_id,
                        "exerciseType": current_exercise,
                        "reps": reps,
                        "stage": stage,
                        "angle": float(angle) if (angle is not None and not math.isnan(angle)) else 0,                        "formScore": safe_form_score(angle),
                        "mistakes": [],
                        "ts": now_ms,
                        "skeleton": keypoints.tolist() if keypoints is not None else [],
                    }

                    if RUN_MODE == "socketio" and sio.connected:
                        try: 
                            sio.emit("ai:progress", payload)
                        except Exception as e: print("[AI] emit progress failed:", e)
                    else:
                        if reps != last_printed_reps:
                            last_printed_reps = reps
                            print(f"[AI] {current_exercise}: reps={reps}, stage={stage}")

                if SHOW_CAMERA:
                    cv2.putText(frame, f"Workout: {current_exercise}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Reps: {reps} | Stage: {stage}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("AI Debug Camera", frame)

            # Press Q to exit the while loop completely
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[AI] Stopping...")
    finally:
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        if RUN_MODE == "socketio":
            try: sio.disconnect()
            except Exception: pass

if __name__ == "__main__":
    main()