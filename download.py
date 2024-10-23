import cv2
import pyttsx3
import time

# Flags to track liveness detection status
haar_detected = False

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def haar_liveness_detection():
    global haar_detected
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    # Reduce resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # You can set lower values like 320 for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera warm-up (optional)
    time.sleep(1)  # Allow camera to warm up

    blink_counter = 0
    consecutive_frames_no_eyes = 0  # Track consecutive frames without eyes
    consecutive_frames_with_eyes = 0  # Track consecutive frames with eyes
    previous_face_position = None
    movement_frames_required = 3
    blink_frames_required = 3  # Adjust the number of frames to count as a blink
    consecutive_movement_frames = 0

    speak("Please nod your head for detection.")
    head_nod_detected = False
    start_time = None
    nod_timeout = 10  # 10 seconds to detect the nod

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi_gray)

                # Debugging output for number of eyes detected
                print(f"Eyes detected: {len(eyes)}")

                # Eye Blink Detection
                if len(eyes) == 0:
                    consecutive_frames_no_eyes += 1
                    consecutive_frames_with_eyes = 0  # Reset when no eyes are detected
                else:
                    consecutive_frames_no_eyes = 0  # Reset if eyes are detected
                    consecutive_frames_with_eyes += 1

                # Count a blink only if no eyes are detected for a few frames,
                # then eyes are detected again (a full blink cycle)
                if consecutive_frames_no_eyes >= blink_frames_required and consecutive_frames_with_eyes > 0:
                    blink_counter += 1
                    consecutive_frames_no_eyes = 0  # Reset after counting a blink
                    consecutive_frames_with_eyes = 0  # Reset the cycle
                    print("Blink detected!")

                # Head Movement Detection
                if previous_face_position is not None:
                    prev_x, prev_y = previous_face_position
                    if abs(prev_x - x) > 15 or abs(prev_y - y) > 15:
                        consecutive_movement_frames += 1
                previous_face_position = (x, y)

                # Detect Head Nod
                if not head_nod_detected and consecutive_movement_frames >= movement_frames_required:
                    speak("Head nod detected. Now, please blink your eyes.")
                    head_nod_detected = True
                    start_time = time.time()  # Restart timer after detecting nod
                    print("Head nod detected. Timer started.")
                    blink_counter = 0  # Reset blink counter

                # Detect Eye Blink after Head Nod
                if head_nod_detected and blink_counter > 0:
                    speak(f"{blink_counter} Eye blink detected. You're completely detected!")
                    haar_detected = True
                    break

        # Print the time since the head nod was detected
        if head_nod_detected:
            elapsed_time = time.time() - start_time
            print(f"Time since nod detected: {elapsed_time:.2f} seconds")
        
            # Check for timeout if head nod has been detected
            if elapsed_time > nod_timeout:
                speak("You did not blink your eyes in time. Please try again.")
                break

        # Display the frame
        cv2.imshow('Haar Liveness Detection', frame)

        # Exit condition
        if haar_detected or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    haar_liveness_detection()
