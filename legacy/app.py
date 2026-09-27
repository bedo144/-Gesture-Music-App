import cv2
import mediapipe as mp


# ==========================================
# 1. Initialize MediaPipe
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. Finger Counting Function
# ==========================================

def count_fingers(hand_landmarks, hand_label):

    landmarks = hand_landmarks.landmark

    fingers = 0


    # --------------------------------------
    # Thumb
    # --------------------------------------

    if hand_label == "Right":

        if landmarks[4].x < landmarks[3].x:
            fingers += 1

    else:

        if landmarks[4].x > landmarks[3].x:
            fingers += 1


    # --------------------------------------
    # Index Finger
    # --------------------------------------

    if landmarks[8].y < landmarks[6].y:
        fingers += 1


    # --------------------------------------
    # Middle Finger
    # --------------------------------------

    if landmarks[12].y < landmarks[10].y:
        fingers += 1


    # --------------------------------------
    # Ring Finger
    # --------------------------------------

    if landmarks[16].y < landmarks[14].y:
        fingers += 1


    # --------------------------------------
    # Pinky Finger
    # --------------------------------------

    if landmarks[20].y < landmarks[18].y:
        fingers += 1


    return fingers


# ==========================================
# 3. Start Camera
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print("=" * 50)
print("HandNote AI")
print("=" * 50)
print("Camera started successfully!")
print("Show your hands to the camera.")
print("Press Q to quit.")
print("=" * 50)


# ==========================================
# 4. Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hands
    results = hands.process(rgb_frame)


    # ======================================
    # 5. Process Hands
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # --------------------------------
            # Left / Right
            # --------------------------------

            hand_label = handedness.classification[0].label

            confidence = handedness.classification[0].score


            # --------------------------------
            # Count fingers
            # --------------------------------

            finger_count = count_fingers(
                hand_landmarks,
                hand_label
            )


            # --------------------------------
            # Draw landmarks
            # --------------------------------

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


            # --------------------------------
            # Wrist position
            # --------------------------------

            wrist = hand_landmarks.landmark[0]

            h, w, _ = frame.shape

            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)


            # --------------------------------
            # Hand label
            # --------------------------------

            hand_text = f"{hand_label} Hand"

            cv2.putText(
                frame,
                hand_text,
                (wrist_x - 60, wrist_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Finger count
            # --------------------------------

            finger_text = f"Fingers: {finger_count}"

            cv2.putText(
                frame,
                finger_text,
                (wrist_x - 60, wrist_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    # ======================================
    # 6. Project Title
    # ======================================

    cv2.putText(
        frame,
        "HandNote AI",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Left = Note | Right = Modifier",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # 7. Show Frame
    # ======================================

    cv2.imshow(
        "HandNote AI",
        frame
    )


    # ======================================
    # 8. Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 9. Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("HandNote AI stopped.")