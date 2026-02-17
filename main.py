import cv2
import numpy as np
import base64
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# استدعاء الملفات من مشروعك
from exercise_counters import ExerciseCounter
from core.rtmpose_processor import RTMPoseProcessor

app = FastAPI()

# السماح بالاتصال من أي مكان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def base64_to_image(base64_string):
    """تحويل النص المشفر إلى صورة"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.websocket("/ws/exercise/{exercise_type}")
async def exercise_endpoint(websocket: WebSocket, exercise_type: str):
    await websocket.accept()
    print(f"Client connected for: {exercise_type}")
    
    # 1. تجهيز العداد والموديل
    counter = ExerciseCounter()
    processor = RTMPoseProcessor(exercise_counter=counter, mode='lightweight') # يمكنك تغيير balanced لـ lightweight للسرعة

    try:
        while True:
            # 2. استقبال البيانات
            data = await websocket.receive_text()
            
            # محاولة استخراج الصورة
            try:
                json_data = json.loads(data)
                image_data = json_data.get("image", "")
            except:
                image_data = data
            
            frame = base64_to_image(image_data)
            
            if frame is not None:
                # 3. المعالجة والحساب
                # الدالة process_frame تقوم بتحديث العداد داخلياً
                _, current_angle, _, keypoints = processor.process_frame(frame, exercise_type)
                
                # 4. إرسال الرد
                response = {
                    "exercise": exercise_type,
                    "reps": counter.counter,
                    "stage": counter.stage,
                    "angle": current_angle if current_angle else 0,
                    "feedback": "Keep going"
                }
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except RuntimeError:
            pass # لو مقفول أصلاً، طنش ومطلعش إيرور

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)