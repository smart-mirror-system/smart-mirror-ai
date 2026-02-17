import cv2
import asyncio
import websockets
import json
import base64
import numpy as np

# إعدادات الاتصال
print("Choose exercise: pushup, squat, bicep_curl, situp, ...")
exercise_name = input("Enter exercise name: ").strip()

# لو ضغطت Enter بدون كتابة، سيفترض أنه pushup
if not exercise_name:
    exercise_name = "pushup"

SERVER_URL = f"ws://localhost:8000/ws/exercise/{exercise_name}"
async def start_camera_stream():
    # 1. فتح الكاميرا
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Connecting to Server: {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected! Starting stream...")
            
            while True:
                # 2. قراءة الصورة من الكاميرا
                ret, frame = cap.read()
                if not ret:
                    break
                
                # === التعديل هنا (انسخ السطرين دول) ===
                # تصغير الصورة إلى ربع الحجم (320x240) وده كافي جداً للذكاء الاصطناعي
                frame = cv2.resize(frame, (320, 240))
                # ====================================

                # 3. تشفير الصورة لإرسالها
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # 4. إرسال الصورة للسيرفر
                data = {"image": jpg_as_text}
                await websocket.send(json.dumps(data))
                
                # 5. استقبال النتيجة من السيرفر
                response = await websocket.recv()
                result = json.loads(response)
                
                # طباعة البيانات في التيرمينال
                # print(result)
                
                # 6. رسم النتيجة على الفيديو (عشان تشوف بعينك)
                count = result.get("reps", 0)
                stage = result.get("stage", "-")
                angle = result.get("angle", 0)
                
                # الكتابة على الشاشة
                cv2.putText(frame, f"Count: {count}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {stage}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Gym AI Test Client', frame)
                
                # الخروج بزر 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure main.py is running first!")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(start_camera_stream())