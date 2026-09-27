import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from collections import deque
from src.preprocessing import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_MODEL_FILE = "left_hand_model.keras"
RIGHT_MODEL_FILE = "right_hand_model.keras"

CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES = 3

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


class AIEngine:

    def __init__(self):

        print("Initializing AI Engine...")

        # ----------------------------------------------------
        # Load Models
        # ----------------------------------------------------

        print("Loading Left Hand model...")

        self.left_model = tf.keras.models.load_model(
            LEFT_MODEL_FILE
        )

        print("Loading Right Hand model...")

        self.right_model = tf.keras.models.load_model(
            RIGHT_MODEL_FILE
        )

        print("AI models loaded successfully.")


        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = (
            mp.solutions.drawing_styles
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open camera."
            )


        # ----------------------------------------------------
        # Gesture Stabilization
        # ----------------------------------------------------

        self.left_history = deque(
            maxlen=STABLE_FRAMES
        )

        self.right_history = deque(
            maxlen=STABLE_FRAMES
        )

        self.last_stable_left = None
        self.last_stable_right = None


        # ----------------------------------------------------
        # Current State
        # ----------------------------------------------------

        self.left_gesture = None
        self.right_gesture = None

        self.left_confidence = 0.0
        self.right_confidence = 0.0

        self.current_output = None


        print("AI Engine ready.")


    # ========================================================
    # NORMALIZE / PREDICT
    # ========================================================

    def predict_hand(
        self,
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

        return classes[index], confidence


    # ========================================================
    # STABILIZATION
    # ========================================================

    def stabilize(
        self,
        prediction,
        history,
        previous
    ):

        if prediction is None:

            history.clear()

            return None

        history.append(
            prediction
        )

        if len(history) < STABLE_FRAMES:

            return previous

        counts = {}

        for gesture in history:

            counts[gesture] = (
                counts.get(gesture, 0) + 1
            )

        best_gesture = max(
            counts,
            key=counts.get
        )

        best_count = counts[
            best_gesture
        ]

        if best_count >= STABLE_FRAMES - 1:

            return best_gesture

        return previous


    # ========================================================
    # GET FRAME
    # ========================================================

    def get_frame(self):

        success, frame = self.cap.read()

        if not success:

            return None

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(
            rgb_frame
        )


        # ----------------------------------------------------
        # Raw predictions
        # ----------------------------------------------------

        raw_left = None
        raw_right = None

        raw_left_confidence = 0.0
        raw_right_confidence = 0.0


        # ----------------------------------------------------
        # Process hands
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            for (
                hand_landmarks,
                handedness
            ) in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_label = (
                    handedness
                    .classification[0]
                    .label
                )


                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    self.mp_drawing_styles
                    .get_default_hand_connections_style()
                )


                # ------------------------------------------------
                # LEFT HAND
                # ------------------------------------------------

                if hand_label == "Left":

                    prediction, confidence = (
                        self.predict_hand(
                            hand_landmarks,
                            self.left_model,
                            LEFT_CLASSES
                        )
                    )

                    raw_left = prediction

                    raw_left_confidence = (
                        confidence
                    )


                # ------------------------------------------------
                # RIGHT HAND
                # ------------------------------------------------

                elif hand_label == "Right":

                    prediction, confidence = (
                        self.predict_hand(
                            hand_landmarks,
                            self.right_model,
                            RIGHT_CLASSES
                        )
                    )

                    raw_right = prediction

                    raw_right_confidence = (
                        confidence
                    )


        # ====================================================
        # STABILIZE
        # ====================================================

        new_left = self.stabilize(
            raw_left,
            self.left_history,
            self.last_stable_left
        )

        if new_left != self.last_stable_left:

            self.last_stable_left = new_left


        new_right = self.stabilize(
            raw_right,
            self.right_history,
            self.last_stable_right
        )

        if new_right != self.last_stable_right:

            self.last_stable_right = new_right


        # ====================================================
        # UPDATE STATE
        # ====================================================

        self.left_gesture = (
            self.last_stable_left
        )

        self.right_gesture = (
            self.last_stable_right
        )

        self.left_confidence = (
            raw_left_confidence
        )

        self.right_confidence = (
            raw_right_confidence
        )


        # ====================================================
        # MUSICAL OUTPUT
        # ====================================================

        self.current_output = (
            self.get_musical_output()
        )


        # ====================================================
        # RETURN DATA
        # ====================================================

        data = {

            "left": self.left_gesture,

            "right": self.right_gesture,

            "left_confidence":
                self.left_confidence,

            "right_confidence":
                self.right_confidence,

            "output":
                self.current_output
        }


        return frame, data


    # ========================================================
    # MUSICAL OUTPUT
    # ========================================================

    def get_musical_output(self):

        note = self.left_gesture
        modifier = self.right_gesture


        if note is None:
            return None

        if modifier is None:
            return None

        if note == "Unknown":
            return None

        if modifier == "Unknown":
            return None


        # ----------------------------------------------------
        # Natural
        # ----------------------------------------------------

        if modifier == "Natural":

            return note


        # ----------------------------------------------------
        # Sharp
        # ----------------------------------------------------

        sharp_mapping = {

            "A": "A#",
            "B": "C",
            "C": "C#",
            "D": "D#",
            "E": "F",
            "F": "F#",
            "G": "G#"
        }


        if modifier == "Sharp":

            return sharp_mapping.get(
                note,
                note
            )


        # ----------------------------------------------------
        # Flat
        # ----------------------------------------------------

        flat_mapping = {

            "A": "G#",
            "B": "A#",
            "C": "B",
            "D": "C#",
            "E": "D#",
            "F": "E",
            "G": "F#"
        }


        if modifier == "Flat":

            return flat_mapping.get(
                note,
                note
            )


        # ----------------------------------------------------
        # Major / Minor
        # ----------------------------------------------------

        if modifier == "Major":

            return f"{note} Major"


        if modifier == "Minor":

            return f"{note} Minor"


        return None


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.left_history.clear()

        self.right_history.clear()

        self.last_stable_left = None
        self.last_stable_right = None

        self.left_gesture = None
        self.right_gesture = None

        self.left_confidence = 0.0
        self.right_confidence = 0.0

        self.current_output = None


    # ========================================================
    # RELEASE
    # ========================================================

    def release(self):

        if self.cap:

            self.cap.release()

        if self.hands:

            self.hands.close()

        print("AI Engine released.")