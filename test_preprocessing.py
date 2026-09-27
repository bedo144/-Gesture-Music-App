import cv2
import mediapipe as mp

from src.preprocessing import normalize_landmarks


# ==========================================
# Initialize MediaPipe
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# Start Camera
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


print("=" * 50)
print("Testing HandNote AI Preprocessing")
print("=" * 50)
print("Show your hand to the camera.")
print("Press Q to quit.")
print("=" * 50)


tested = False


# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # ======================================
    # Test preprocessing
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            features = normalize_landmarks(
                hand_landmarks
            )

            if not tested:

                print("\nPreprocessing Test")
                print("-" * 30)

                print(
                    "Feature shape:",
                    features.shape
                )

                print(
                    "Number of features:",
                    len(features)
                )

                print(
                    "First 9 values:",
                    features[:9]
                )

                print("-" * 30)

                tested = True


    cv2.imshow(
        "Preprocessing Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()