#  Gesture Music App (HandNoteAI)

An AI-powered music control application that uses computer vision and deep learning models to translate hand gestures into real-time sound and music commands.

---

##  Features

* **Real-time Gesture Recognition:** Uses trained Keras models (`left_hand_model.keras` and `right_hand_model.keras`) to detect hand poses via camera feed.
* **Interactive Sound Engine:** Triggers audio feedback and music controls based on specific hand movements.
* **Dataset Management:** Includes scripts for dataset collection, preprocessing, and model verification.

---

##  Project Structure

```text
├── data/                    # Dataset storage
├── src/                     # Core application source code
├── legacy/                  # Legacy code and previous iterations
├── ai_engine.py             # Main AI engine handler
├── sound_engine.py          # Audio playback and synthesizer logic
├── handnote_app.py          # Main application launcher
├── realtime_inference.py    # Real-time hand tracking & prediction engine
├── data_collector.py        # Script for recording gesture training data
├── train_left_model.py      # Training script for left-hand gestures
├── train_right_model.py     # Training script for right-hand gestures
└── requirements.txt         # Project dependencies
