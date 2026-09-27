import cv2
import mediapipe as mp
import csv
import os
import time

from src.preprocessing import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = "data"

DATASET_FILE = os.path.join(
    DATA_FOLDER,
    "handnote_dataset.csv"
)

# Target number of samples for the selected class
TARGET_SAMPLES = 120

# Automatic collection settings
AUTO_CAPTURE_DELAY = 0.25


# ============================================================
# CREATE DATA FOLDER
# ============================================================

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = (
    mp.solutions.drawing_styles
)


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# DATASET CLASSES
# ============================================================

NOTE_CLASSES = [

    "C",
    "D",
    "E",
    "F",
    "G",
    "A",
    "B"

]


MODIFIER_CLASSES = [

    "Natural",
    "Sharp",
    "Flat",
    "Major",
    "Minor"

]


# ============================================================
# SELECT HAND
# ============================================================

print("=" * 60)

print("HandNote AI - Dataset Collector")

print("=" * 60)


print("\nWhich hand are you collecting?")

print("1 - Left Hand (Notes)")

print("2 - Right Hand (Modifiers)")


hand_choice = input(
    "\nEnter 1 or 2: "
).strip()


# ============================================================
# LEFT HAND
# ============================================================

if hand_choice == "1":

    target_hand = "Left"

    print("\nAvailable Notes:")

    for i, note in enumerate(
        NOTE_CLASSES,
        start=1
    ):

        print(
            f"{i} - {note}"
        )


    class_choice = input(
        "\nChoose the note number: "
    ).strip()


    try:

        target_class = NOTE_CLASSES[
            int(class_choice) - 1
        ]

    except (
        ValueError,
        IndexError
    ):

        print(
            "Invalid choice."
        )

        exit()


# ============================================================
# RIGHT HAND
# ============================================================

elif hand_choice == "2":

    target_hand = "Right"

    print("\nAvailable Modifiers:")

    for i, modifier in enumerate(
        MODIFIER_CLASSES,
        start=1
    ):

        print(
            f"{i} - {modifier}"
        )


    class_choice = input(
        "\nChoose the modifier number: "
    ).strip()


    try:

        target_class = MODIFIER_CLASSES[
            int(class_choice) - 1
        ]

    except (
        ValueError,
        IndexError
    ):

        print(
            "Invalid choice."
        )

        exit()


else:

    print(
        "Invalid hand choice."
    )

    exit()


# ============================================================
# CREATE CSV IF IT DOES NOT EXIST
# ============================================================

file_exists = os.path.exists(
    DATASET_FILE
)


if not file_exists:

    header = [

        "hand_side",

        "label"

    ]


    for i in range(1, 64):

        header.append(
            f"feature_{i}"
        )


    with open(
        DATASET_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            header
        )


# ============================================================
# COUNT EXISTING SAMPLES
# ============================================================

sample_count = 0


with open(
    DATASET_FILE,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(
        file
    )


    for row in reader:

        if (

            row["hand_side"]
            == target_hand

            and

            row["label"]
            == target_class

        ):

            sample_count += 1


# ============================================================
# START CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    hands.close()

    exit()


# ============================================================
# COLLECTION MODE
# ============================================================

print("\n" + "=" * 60)

print("COLLECTION READY")

print("=" * 60)

print(
    f"Hand  : {target_hand}"
)

print(
    f"Class : {target_class}"
)

print(
    f"Existing samples: {sample_count}"
)

print(
    f"Target samples: {TARGET_SAMPLES}"
)

print("\nControls:")

print(
    "SPACE → Manual capture"
)

print(
    "A     → Automatic capture"
)

print(
    "S     → Stop automatic capture"
)

print(
    "Q     → Quit"
)

print("=" * 60)


# ============================================================
# AUTO CAPTURE STATE
# ============================================================

auto_capture = False

last_capture_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Could not read frame."
        )

        break


    # ========================================================
    # MIRROR CAMERA
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # BGR → RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    results = hands.process(
        rgb_frame
    )


    detected_landmarks = None


    # ========================================================
    # FIND TARGET HAND
    # ========================================================

    if results.multi_hand_landmarks:


        for hand_landmarks, handedness in zip(

            results.multi_hand_landmarks,

            results.multi_handedness

        ):


            hand_label = (

                handedness
                .classification[0]
                .label
            )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS,

                mp_drawing_styles
                    .get_default_hand_landmarks_style(),

                mp_drawing_styles
                    .get_default_hand_connections_style()

            )


            # =================================================
            # CHECK TARGET HAND
            # =================================================

            if hand_label == target_hand:

                detected_landmarks = (
                    hand_landmarks
                )


    # ========================================================
    # DISPLAY TITLE
    # ========================================================

    cv2.putText(

        frame,

        "HandNote AI - Data Collector",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )


    # ========================================================
    # DISPLAY TARGET
    # ========================================================

    cv2.putText(

        frame,

        f"Target: {target_hand} - {target_class}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2

    )


    # ========================================================
    # DISPLAY SAMPLE COUNT
    # ========================================================

    cv2.putText(

        frame,

        f"Samples: {sample_count}/{TARGET_SAMPLES}",

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    # ========================================================
    # DISPLAY HAND STATUS
    # ========================================================

    if detected_landmarks is not None:

        cv2.putText(

            frame,

            "HAND DETECTED",

            (20, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )

    else:

        cv2.putText(

            frame,

            f"SHOW YOUR {target_hand} HAND",

            (20, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2

        )


    # ========================================================
    # AUTO CAPTURE STATUS
    # ========================================================

    if auto_capture:

        cv2.putText(

            frame,

            "AUTO CAPTURE: ON",

            (20, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2

        )

    else:

        cv2.putText(

            frame,

            "AUTO CAPTURE: OFF",

            (20, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "HandNote AI Dataset Collector",

        frame

    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # START AUTO CAPTURE
    # ========================================================

    if key == ord("a"):

        auto_capture = True

        print(
            "\n▶ Automatic collection started."
        )


    # ========================================================
    # STOP AUTO CAPTURE
    # ========================================================

    elif key == ord("s"):

        auto_capture = False

        print(
            "\n⏸ Automatic collection stopped."
        )


    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        break


    # ========================================================
    # MANUAL CAPTURE
    # ========================================================

    elif key == ord(" "):

        if detected_landmarks is not None:

            features = normalize_landmarks(

                detected_landmarks

            )


            row = [

                target_hand,

                target_class

            ]


            row.extend(

                features.tolist()

            )


            with open(

                DATASET_FILE,

                "a",

                newline=""

            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    row
                )


            sample_count += 1


            print(

                f"Sample {sample_count} saved "
                f"for {target_hand} - {target_class}"

            )


        else:

            print(

                f"No {target_hand} hand detected."

            )


    # ========================================================
    # AUTOMATIC CAPTURE
    # ========================================================

    if auto_capture:

        current_time = time.time()


        if (

            current_time
            - last_capture_time
            >= AUTO_CAPTURE_DELAY

        ):


            if detected_landmarks is not None:


                features = normalize_landmarks(

                    detected_landmarks

                )


                row = [

                    target_hand,

                    target_class

                ]


                row.extend(

                    features.tolist()

                )


                with open(

                    DATASET_FILE,

                    "a",

                    newline=""

                ) as file:

                    writer = csv.writer(
                        file
                    )

                    writer.writerow(
                        row
                    )


                sample_count += 1


                last_capture_time = (
                    current_time
                )


                print(

                    f"Auto sample "
                    f"{sample_count} saved "
                    f"for {target_hand} - "
                    f"{target_class}"

                )


                # ==========================================
                # TARGET REACHED
                # ==========================================

                if sample_count >= TARGET_SAMPLES:

                    auto_capture = False

                    print(
                        "\n🎉 Target reached!"
                    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


print(
    "\nDataset collection stopped."
)

print(
    f"Total samples for this class: "
    f"{sample_count}"
)

print(
    f"Dataset file: {DATASET_FILE}"
)