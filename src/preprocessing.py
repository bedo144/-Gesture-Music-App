import numpy as np


def normalize_landmarks(hand_landmarks):
    """
    Convert MediaPipe hand landmarks
    into normalized relative coordinates.

    Steps:
    1. Move wrist to the origin.
    2. Normalize the scale of the hand.
    3. Return 63 features.
    """

    # ==========================================
    # 1. Get wrist
    # ==========================================

    wrist = hand_landmarks.landmark[0]

    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z


    # ==========================================
    # 2. Create relative coordinates
    # ==========================================

    relative_landmarks = []

    for landmark in hand_landmarks.landmark:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        relative_landmarks.append([x, y, z])


    # ==========================================
    # 3. Convert to NumPy array
    # ==========================================

    relative_landmarks = np.array(
        relative_landmarks,
        dtype=np.float32
    )


    # ==========================================
    # 4. Calculate hand scale
    # ==========================================

    distances = np.linalg.norm(
        relative_landmarks,
        axis=1
    )

    scale = np.max(distances)


    # ==========================================
    # 5. Avoid division by zero
    # ==========================================

    if scale > 0:

        relative_landmarks = (
            relative_landmarks / scale
        )


    # ==========================================
    # 6. Flatten to 63 features
    # ==========================================

    features = relative_landmarks.flatten()


    return features