import cv2
import mediapipe as mp
import numpy as np
import base64
import asyncio
import time
from scipy.spatial import distance
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# Setup MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Setup OpenCV Video Capture (ensure no other app uses the webcam)
cap = cv2.VideoCapture(0)

# FPS counter variables
fps_counter = 0
last_time = time.time()

# Drawing specifications for the mesh (only the wireframe, no dots)
drawing_spec_mesh = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)  # No dots
drawing_spec_contours = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)  # No dots

# Blink and yawn counters and thresholds
blink_count = 0
yawn_count = 0
blink_start_time = time.time()
previous_ear = 1.0
previous_mar = 0.0
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
blink_reset_time = 60  # seconds
time_counter_start = time.time()

# Function to compute EAR (Eye Aspect Ratio)
def compute_EAR(top, bottom, left, right):
    vertical_dist = distance.euclidean(top, bottom)
    horizontal_dist = distance.euclidean(left, right)
    return vertical_dist / horizontal_dist

# Function to compute MAR (Mouth Aspect Ratio)
def compute_MAR(top, bottom, left, right):
    vertical_dist = distance.euclidean(top, bottom)
    horizontal_dist = distance.euclidean(left, right)
    return vertical_dist / horizontal_dist

@app.get("/")
async def get():
    # Serve the index.html from the "static" folder.
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    global fps_counter, last_time, time_counter_start, blink_count, yawn_count, previous_ear, previous_mar

    # Initialize the FaceMesh model once for efficiency
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, image = cap.read()
            if not ret:
                continue

            # Increment FPS counter and update every second
            current_time = time.time()
            fps_counter += 1
            if current_time - last_time >= 1:
                fps = fps_counter
                fps_counter = 0
                last_time = current_time
                cv2.putText(image, f'FPS: {fps}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Resize the image to a consistent size
            image = cv2.resize(image, (640, 480))

            # Convert to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            # Convert back to BGR for further OpenCV processing
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # If face landmarks are detected, process them
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw only the wireframe (connections), no dots
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        connection_drawing_spec=drawing_spec_mesh,
                        landmark_drawing_spec=None  # Ensure no dots are drawn
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        connection_drawing_spec=drawing_spec_contours,
                        landmark_drawing_spec=None  # Ensure no dots are drawn
                    )

                    # Get the facial landmarks and image dimensions
                    landmarks = face_landmarks.landmark
                    image_h, image_w, _ = image.shape

                    def get_point(index):
                        return int(landmarks[index].x * image_w), int(landmarks[index].y * image_h)

                    # Extract points for eyes and mouth
                    left_eye_top = get_point(159)
                    left_eye_bottom = get_point(23)
                    left_eye_left = get_point(130)
                    left_eye_right = get_point(243)
                    right_eye_top = get_point(386)
                    right_eye_bottom = get_point(253)
                    right_eye_left = get_point(463)
                    right_eye_right = get_point(359)
                    mouth_top = get_point(13)
                    mouth_bottom = get_point(14)
                    mouth_left = get_point(78)
                    mouth_right = get_point(308)

                    # Compute Eye Aspect Ratio (EAR)
                    left_EAR = compute_EAR(left_eye_top, left_eye_bottom, left_eye_left, left_eye_right)
                    right_EAR = compute_EAR(right_eye_top, right_eye_bottom, right_eye_left, right_eye_right)
                    avg_EAR = (left_EAR + right_EAR) / 2

                    # Compute Mouth Aspect Ratio (MAR)
                    MAR = compute_MAR(mouth_top, mouth_bottom, mouth_left, mouth_right)

                    # Drowsiness detection (unused variable drowsy here)
                    drowsy = avg_EAR < 0.25

                    # Blink detection
                    if avg_EAR < EAR_THRESHOLD and previous_ear >= EAR_THRESHOLD:
                        blink_count += 1
                        blink_start_time = time.time()
                    previous_ear = avg_EAR

                    # Yawn detection
                    if MAR > MAR_THRESHOLD and previous_mar <= MAR_THRESHOLD:
                        yawn_count += 1
                    previous_mar = MAR

                    # Reset blink and yawn counts every blink_reset_time seconds
                    elapsed_time = time.time() - time_counter_start
                    if elapsed_time >= blink_reset_time:
                        blink_count = 0
                        yawn_count = 0
                        time_counter_start = time.time()

                    # Compute head tilt (Pitch & Roll)
                    nose_tip = get_point(1)
                    chin = get_point(152)
                    left_ear = get_point(234)
                    right_ear = get_point(454)
                    delta_y = chin[1] - nose_tip[1]
                    delta_x = right_ear[0] - left_ear[0]
                    pitch = np.degrees(np.arctan(delta_y / (image_h / 2)))
                    roll = np.degrees(np.arctan(delta_x / (image_w / 2)))

                    # Flip the image for a mirror effect
                    flipped_image = cv2.flip(image, 1)

                    # Encode the processed frame as JPEG then base64
                    _, buffer = cv2.imencode('.jpg', flipped_image)
                    frame_data = base64.b64encode(buffer).decode('utf-8')

                    # Create a message with the frame and the stats:
                    # Format: base64_frame_data;avg_EAR,pitch,roll,blink_count,yawn_count
                    message = f"{frame_data};{avg_EAR:.2f},{pitch:.2f},{roll:.2f},{blink_count},{yawn_count}"
                    await websocket.send_text(message)
            else:
                # If no face is detected, send the flipped frame with dummy stats
                _, buffer = cv2.imencode('.jpg', cv2.flip(image, 1))
                frame_data = base64.b64encode(buffer).decode('utf-8')
                message = f"{frame_data};0.00,0.00,0.00,0,0"
                await websocket.send_text(message)

@app.on_event("shutdown")
async def shutdown():
    cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)