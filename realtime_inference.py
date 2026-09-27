import os

# Reduce TensorFlow console messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from collections import deque
from sound_engine import SoundEngine
from src.preprocessing import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_MODEL_FILE = "left_hand_model.keras"
RIGHT_MODEL_FILE = "right_hand_model.keras"

CONFIDENCE_THRESHOLD = 0.60

# Number of frames used for gesture stabilization
STABLE_FRAMES = 3

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# CLASSES
# ============================================================

LEFT_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G"
]

RIGHT_CLASSES = [
    "Flat",
    "Major",
    "Minor",
    "Natural",
    "Sharp"
]


# ============================================================
# GESTURE HISTORY
# ============================================================

left_gesture_history = deque(
    maxlen=STABLE_FRAMES
)

right_gesture_history = deque(
    maxlen=STABLE_FRAMES
)

last_stable_left = None
last_stable_right = None


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("HandNote AI - Real-Time AI System")
print("=" * 60)

print("\nLoading AI models...")

left_model = tf.keras.models.load_model(
    LEFT_MODEL_FILE
)

right_model = tf.keras.models.load_model(
    RIGHT_MODEL_FILE
)

print("Left Hand Model  : Loaded")
print("Right Hand Model : Loaded")


# ============================================================
# SOUND ENGINE
# ============================================================

print("\nLoading sound engine...")

sound_engine = SoundEngine()

sound_enabled = True

print("Sound Engine      : Ready")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# MUSICAL MAPPING
# ============================================================

def get_musical_note(note, modifier):

    if modifier == "Natural":
        return note

    sharp_mapping = {

        "A": "A#",
        "B": "C",
        "C": "C#",
        "D": "D#",
        "E": "F",
        "F": "F#",
        "G": "G#"
    }

    flat_mapping = {

        "A": "G#",
        "B": "A#",
        "C": "B",
        "D": "C#",
        "E": "D#",
        "F": "E",
        "G": "F#"
    }

    if modifier == "Sharp":
        return sharp_mapping.get(
            note,
            note
        )

    if modifier == "Flat":
        return flat_mapping.get(
            note,
            note
        )

    return note


# ============================================================
# GESTURE STABILIZATION
# ============================================================

def stabilize_gesture(
    prediction,
    history,
    last_stable
):

    if prediction is None:

        history.clear()

        return None

    history.append(prediction)

    if len(history) < STABLE_FRAMES:

        return last_stable

    counts = {}

    for gesture in history:

        counts[gesture] = (
            counts.get(gesture, 0) + 1
        )

    best_gesture = max(
        counts,
        key=counts.get
    )

    best_count = counts[best_gesture]

    if best_count >= STABLE_FRAMES - 1:

        return best_gesture

    return last_stable


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_hand(
    hand_landmarks,
    model,
    classes
):

    features = normalize_landmarks(
        hand_landmarks
    )

    features = np.expand_dims(
        features,
        axis=0
    )

    predictions = model.predict(
        features,
        verbose=0
    )[0]

    index = np.argmax(
        predictions
    )

    confidence = float(
        predictions[index]
    )

    if confidence < CONFIDENCE_THRESHOLD:

        return "Unknown", confidence

    label = classes[index]

    return label, confidence


# ============================================================
# PLAY MUSICAL OUTPUT
# ============================================================

last_played_gesture = None


def play_musical_output(
    note,
    modifier
):

    global last_played_gesture

    if note is None:
        return

    if modifier is None:
        return

    if note == "Unknown":
        return

    if modifier == "Unknown":
        return

    gesture_id = (
        f"{note}_{modifier}"
    )

    # Don't replay the same gesture
    if gesture_id == last_played_gesture:

        return

    last_played_gesture = gesture_id

    if not sound_enabled:

        return

    # Major / Minor = chord
    if modifier == "Major":

        sound_engine.play_chord(
            note,
            "Major"
        )

        return

    if modifier == "Minor":

        sound_engine.play_chord(
            note,
            "Minor"
        )

        return

    # Natural / Sharp / Flat = single note

    musical_note = get_musical_note(
        note,
        modifier
    )

    sound_engine.play_note(
        musical_note
    )


# ============================================================
# DRAW TEXT HELPER
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

if not cap.isOpened():

    print("\nERROR: Could not open camera.")

    sound_engine.cleanup()
    hands.close()

    exit()


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = 0

fps = 0


# ============================================================
# CURRENT STATE
# ============================================================

current_left = None
current_right = None

left_confidence = 0.0
right_confidence = 0.0

current_note = None
current_modifier = None
current_output = None


print("\n" + "=" * 60)
print("REAL-TIME SYSTEM STARTED")
print("=" * 60)

print("\nControls:")
print("S → Toggle sound ON/OFF")
print("R → Reset current gesture")
print("Q → Quit")

print("\nHand Mapping:")
print("Left Hand  → Musical Note")
print("Right Hand → Modifier")

print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "ERROR: Could not read frame."
        )

        break

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb_frame
    )


    # ========================================================
    # RAW PREDICTIONS
    # ========================================================

    raw_left = None
    raw_right = None

    raw_left_confidence = 0.0
    raw_right_confidence = 0.0


    # ========================================================
    # DETECT HANDS
    # ========================================================

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_label = (
                handedness.classification[0].label
            )

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


            # =================================================
            # LEFT HAND
            # =================================================

            if hand_label == "Left":

                prediction, confidence = predict_hand(
                    hand_landmarks,
                    left_model,
                    LEFT_CLASSES
                )

                raw_left = prediction

                raw_left_confidence = confidence


            # =================================================
            # RIGHT HAND
            # =================================================

            elif hand_label == "Right":

                prediction, confidence = predict_hand(
                    hand_landmarks,
                    right_model,
                    RIGHT_CLASSES
                )

                raw_right = prediction

                raw_right_confidence = confidence


    # ========================================================
    # STABILIZE LEFT
    # ========================================================

    new_left = stabilize_gesture(
        raw_left,
        left_gesture_history,
        last_stable_left
    )

    if new_left != last_stable_left:

        last_stable_left = new_left


    # ========================================================
    # STABILIZE RIGHT
    # ========================================================

    new_right = stabilize_gesture(
        raw_right,
        right_gesture_history,
        last_stable_right
    )

    if new_right != last_stable_right:

        last_stable_right = new_right


    # ========================================================
    # CURRENT STATE
    # ========================================================

    current_left = last_stable_left

    current_right = last_stable_right

    left_confidence = raw_left_confidence

    right_confidence = raw_right_confidence


    # ========================================================
    # MUSICAL OUTPUT
    # ========================================================

    if (
        current_left is not None
        and current_right is not None
        and current_left != "Unknown"
        and current_right != "Unknown"
    ):

        current_note = current_left

        current_modifier = current_right

        if current_modifier in [
            "Natural",
            "Sharp",
            "Flat"
        ]:

            current_output = get_musical_note(
                current_note,
                current_modifier
            )

        else:

            current_output = (
                f"{current_note} "
                f"{current_modifier}"
            )

        play_musical_output(
            current_note,
            current_modifier
        )

    else:

        current_note = None
        current_modifier = None
        current_output = None

        last_played_gesture = None


    # ========================================================
    # FPS
    # ========================================================

    current_time = cv2.getTickCount()

    if previous_time != 0:

        time_difference = (
            current_time - previous_time
        ) / cv2.getTickFrequency()

        if time_difference > 0:

            fps = 1 / time_difference

    previous_time = current_time


    # ========================================================
    # UI PANEL
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (420, 220),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )


    # ========================================================
    # TITLE
    # ========================================================

    draw_text(
        frame,
        "HANDNOTE AI",
        (20, 35),
        (0, 255, 255),
        0.9,
        2
    )


    # ========================================================
    # FPS
    # ========================================================

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (300, 35),
        (255, 255, 255),
        0.6,
        2
    )


    # ========================================================
    # LEFT HAND INFO
    # ========================================================

    left_display = (
        current_left
        if current_left is not None
        else "---"
    )

    draw_text(
        frame,
        f"LEFT  NOTE: {left_display}",
        (20, 75),
        (0, 255, 0),
        0.7,
        2
    )

    if left_confidence > 0:

        draw_text(
            frame,
            f"Confidence: {left_confidence * 100:.1f}%",
            (220, 75),
            (255, 255, 255),
            0.5,
            1
        )


    # ========================================================
    # RIGHT HAND INFO
    # ========================================================

    right_display = (
        current_right
        if current_right is not None
        else "---"
    )

    draw_text(
        frame,
        f"RIGHT: {right_display}",
        (20, 110),
        (0, 200, 255),
        0.7,
        2
    )

    if right_confidence > 0:

        draw_text(
            frame,
            f"Confidence: {right_confidence * 100:.1f}%",
            (220, 110),
            (255, 255, 255),
            0.5,
            1
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    output_display = (
        current_output
        if current_output is not None
        else "---"
    )

    draw_text(
        frame,
        f"OUTPUT: {output_display}",
        (20, 155),
        (255, 0, 255),
        0.8,
        2
    )


    # ========================================================
    # SOUND STATUS
    # ========================================================

    sound_status = (
        "ON"
        if sound_enabled
        else "OFF"
    )

    draw_text(
        frame,
        f"Sound: {sound_status}",
        (20, 190),
        (0, 255, 255),
        0.6,
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    draw_text(
        frame,
        "S: Sound | R: Reset | Q: Quit",
        (20, 215),
        (200, 200, 200),
        0.5,
        1
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "HandNote AI - Real-Time AI",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Toggle sound
    if key == ord("s"):

        sound_enabled = not sound_enabled

        print(
            f"Sound: "
            f"{'ON' if sound_enabled else 'OFF'}"
        )

        # Reset so next gesture can play
        last_played_gesture = None


    # Reset gesture
    elif key == ord("r"):

        left_gesture_history.clear()
        right_gesture_history.clear()

        last_stable_left = None
        last_stable_right = None

        last_played_gesture = None

        current_note = None
        current_modifier = None
        current_output = None

        print("Gesture state reset.")


    # Quit
    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

sound_engine.cleanup()

print("\n" + "=" * 60)
print("HandNote AI stopped.")
print("=" * 60)