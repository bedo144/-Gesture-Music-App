import sys
import time

import cv2

from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal
)

from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QFont
)

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame
)

from ai_engine import AIEngine
from sound_engine import SoundEngine


# ============================================================
# AI WORKER
# ============================================================

class AIWorker(QThread):

    frame_ready = pyqtSignal(object, dict)
    error = pyqtSignal(str)

    def __init__(self):

        super().__init__()

        self.running = True
        self.engine = None

    def run(self):

        try:

            self.engine = AIEngine()

            while self.running:

                result = self.engine.get_frame()

                if result is None:

                    self.error.emit(
                        "Could not read camera."
                    )

                    break

                frame, data = result

                self.frame_ready.emit(
                    frame,
                    data
                )

                time.sleep(0.01)

        except Exception as e:

            self.error.emit(
                str(e)
            )

    def stop(self):

        self.running = False

        if self.engine:

            self.engine.release()

        self.wait()

    def reset(self):

        if self.engine:

            self.engine.reset()


# ============================================================
# MAIN WINDOW
# ============================================================

class HandNoteApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "HandNote AI"
        )

        self.setMinimumSize(
            1200,
            750
        )

        # ----------------------------------------------------
        # Sound
        # ----------------------------------------------------

        self.sound_enabled = True

        self.sound_engine = SoundEngine()

        # Stores the last gesture that produced sound
        self.last_played = None

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.setup_ui()

        # ----------------------------------------------------
        # Start AI Worker
        # ----------------------------------------------------

        self.worker = AIWorker()

        self.worker.frame_ready.connect(
            self.update_frame
        )

        self.worker.error.connect(
            self.show_error
        )

        self.worker.start()

    # ========================================================
    # UI SETUP
    # ========================================================

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            20,
            20,
            20,
            20
        )

        main_layout.setSpacing(
            15
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        title = QLabel(
            "🎵 HANDNOTE AI"
        )

        title.setFont(
            QFont(
                "Segoe UI",
                24,
                QFont.Weight.Bold
            )
        )

        subtitle = QLabel(
            "AI-Powered Gesture Music System"
        )

        subtitle.setFont(
            QFont(
                "Segoe UI",
                10
            )
        )

        title_box = QVBoxLayout()

        title_box.addWidget(
            title
        )

        title_box.addWidget(
            subtitle
        )

        self.status_label = QLabel(
            "● AI STARTING..."
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        self.status_label.setFont(
            QFont(
                "Segoe UI",
                11,
                QFont.Weight.Bold
            )
        )

        header.addLayout(
            title_box
        )

        header.addStretch()

        header.addWidget(
            self.status_label
        )

        main_layout.addLayout(
            header
        )

        # ====================================================
        # CONTENT
        # ====================================================

        content_layout = QHBoxLayout()

        content_layout.setSpacing(
            15
        )

        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_frame = QFrame()

        camera_frame.setFrameShape(
            QFrame.Shape.StyledPanel
        )

        camera_layout = QVBoxLayout(
            camera_frame
        )

        camera_title = QLabel(
            "CAMERA FEED"
        )

        camera_title.setFont(
            QFont(
                "Segoe UI",
                11,
                QFont.Weight.Bold
            )
        )

        self.camera_label = QLabel()

        self.camera_label.setMinimumSize(
            760,
            500
        )

        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.camera_label.setText(
            "Starting camera..."
        )

        camera_layout.addWidget(
            camera_title
        )

        camera_layout.addWidget(
            self.camera_label
        )

        content_layout.addWidget(
            camera_frame,
            stretch=3
        )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        info_panel = QFrame()

        info_panel.setFrameShape(
            QFrame.Shape.StyledPanel
        )

        info_layout = QVBoxLayout(
            info_panel
        )

        # ----------------------------------------------------
        # Left Hand
        # ----------------------------------------------------

        left_title = QLabel(
            "LEFT HAND"
        )

        left_title.setFont(
            QFont(
                "Segoe UI",
                10,
                QFont.Weight.Bold
            )
        )

        self.left_value = QLabel(
            "---"
        )

        self.left_value.setFont(
            QFont(
                "Segoe UI",
                32,
                QFont.Weight.Bold
            )
        )

        self.left_value.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.left_confidence = QLabel(
            "Confidence: --"
        )

        self.left_confidence.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        info_layout.addWidget(
            left_title
        )

        info_layout.addWidget(
            self.left_value
        )

        info_layout.addWidget(
            self.left_confidence
        )

        # ----------------------------------------------------
        # Separator
        # ----------------------------------------------------

        separator1 = QFrame()

        separator1.setFrameShape(
            QFrame.Shape.HLine
        )

        info_layout.addWidget(
            separator1
        )

        # ----------------------------------------------------
        # Right Hand
        # ----------------------------------------------------

        right_title = QLabel(
            "RIGHT HAND"
        )

        right_title.setFont(
            QFont(
                "Segoe UI",
                10,
                QFont.Weight.Bold
            )
        )

        self.right_value = QLabel(
            "---"
        )

        self.right_value.setFont(
            QFont(
                "Segoe UI",
                24,
                QFont.Weight.Bold
            )
        )

        self.right_value.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.right_confidence = QLabel(
            "Confidence: --"
        )

        self.right_confidence.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        info_layout.addWidget(
            right_title
        )

        info_layout.addWidget(
            self.right_value
        )

        info_layout.addWidget(
            self.right_confidence
        )

        # ----------------------------------------------------
        # Separator
        # ----------------------------------------------------

        separator2 = QFrame()

        separator2.setFrameShape(
            QFrame.Shape.HLine
        )

        info_layout.addWidget(
            separator2
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_title = QLabel(
            "CURRENT OUTPUT"
        )

        output_title.setFont(
            QFont(
                "Segoe UI",
                10,
                QFont.Weight.Bold
            )
        )

        self.output_value = QLabel(
            "---"
        )

        self.output_value.setFont(
            QFont(
                "Segoe UI",
                20,
                QFont.Weight.Bold
            )
        )

        self.output_value.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.output_value.setWordWrap(
            True
        )

        info_layout.addWidget(
            output_title
        )

        info_layout.addWidget(
            self.output_value
        )

        info_layout.addStretch()

        content_layout.addWidget(
            info_panel,
            stretch=1
        )

        main_layout.addLayout(
            content_layout
        )

        # ====================================================
        # FOOTER
        # ====================================================

        footer = QHBoxLayout()

        self.sound_button = QPushButton(
            "🔊 Sound: ON"
        )

        self.sound_button.clicked.connect(
            self.toggle_sound
        )

        reset_button = QPushButton(
            "↻ Reset"
        )

        reset_button.clicked.connect(
            self.reset_ai
        )

        exit_button = QPushButton(
            "✕ Exit"
        )

        exit_button.clicked.connect(
            self.close
        )

        footer.addWidget(
            self.sound_button
        )

        footer.addStretch()

        footer.addWidget(
            reset_button
        )

        footer.addWidget(
            exit_button
        )

        main_layout.addLayout(
            footer
        )

        # ====================================================
        # WINDOW STYLE
        # ====================================================

        self.setStyleSheet("""

            QMainWindow {
                background-color: #101218;
            }

            QWidget {
                color: #F1F1F1;
                font-family: "Segoe UI";
            }

            QFrame {
                background-color: #181B22;
                border: 1px solid #292D36;
                border-radius: 12px;
            }

            QLabel {
                background: transparent;
            }

            QPushButton {
                background-color: #242832;
                border: 1px solid #363B47;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 11pt;
            }

            QPushButton:hover {
                background-color: #303541;
            }

        """)

    # ========================================================
    # PLAY SOUND
    # ========================================================

    def play_detected_sound(self, output):

        if not self.sound_enabled:
            return

        if not output:
            return

        # ----------------------------------------------------
        # Do not replay the same gesture continuously
        # ----------------------------------------------------

        if output == self.last_played:
            return

        # Save current output
        self.last_played = output

        try:

            # =================================================
            # Major / Minor chord
            # =================================================

            if output.endswith(" Major"):

                root = output.replace(
                    " Major",
                    ""
                )

                self.sound_engine.play_chord(
                    root,
                    "Major"
                )

            elif output.endswith(" Minor"):

                root = output.replace(
                    " Minor",
                    ""
                )

                self.sound_engine.play_chord(
                    root,
                    "Minor"
                )

            # =================================================
            # Normal note
            # =================================================

            else:

                self.sound_engine.play_note(
                    output
                )

            print(
                f"🔊 Playing: {output}"
            )

        except Exception as e:

            print(
                "Sound error:",
                e
            )

    # ========================================================
    # UPDATE FRAME
    # ========================================================

    def update_frame(
        self,
        frame,
        data
    ):

        # ----------------------------------------------------
        # Convert OpenCV frame → QImage
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = (
            rgb.shape
        )

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            image
        )

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        # ====================================================
        # GET DATA
        # ====================================================

        left = data.get(
            "left"
        )

        right = data.get(
            "right"
        )

        left_conf = data.get(
            "left_confidence",
            0
        )

        right_conf = data.get(
            "right_confidence",
            0
        )

        output = data.get(
            "output"
        )

        # ====================================================
        # LEFT HAND
        # ====================================================

        self.left_value.setText(
            left if left else "---"
        )

        if left_conf > 0:

            self.left_confidence.setText(
                f"Confidence: "
                f"{left_conf * 100:.1f}%"
            )

        else:

            self.left_confidence.setText(
                "Confidence: --"
            )

        # ====================================================
        # RIGHT HAND
        # ====================================================

        self.right_value.setText(
            right if right else "---"
        )

        if right_conf > 0:

            self.right_confidence.setText(
                f"Confidence: "
                f"{right_conf * 100:.1f}%"
            )

        else:

            self.right_confidence.setText(
                "Confidence: --"
            )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.output_value.setText(
            output if output else "---"
        )

        # ====================================================
        # PLAY SOUND
        # ====================================================

        if output:

            self.play_detected_sound(
                output
            )

        else:

            # No valid gesture
            # Allow the same gesture to play again
            # when the user makes it again.

            self.last_played = None

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label.setText(
            "● AI ONLINE"
        )

    # ========================================================
    # ERROR
    # ========================================================

    def show_error(
        self,
        message
    ):

        self.status_label.setText(
            "● AI ERROR"
        )

        print(
            "AI ERROR:",
            message
        )

    # ========================================================
    # TOGGLE SOUND
    # ========================================================

    def toggle_sound(self):

        self.sound_enabled = (
            not self.sound_enabled
        )

        if self.sound_enabled:

            self.sound_button.setText(
                "🔊 Sound: ON"
            )

            # Allow current gesture to play again
            self.last_played = None

        else:

            self.sound_button.setText(
                "🔇 Sound: OFF"
            )

            self.sound_engine.stop()

            self.last_played = None

    # ========================================================
    # RESET
    # ========================================================

    def reset_ai(self):

        if self.worker:

            self.worker.reset()

        self.sound_engine.stop()

        self.left_value.setText(
            "---"
        )

        self.right_value.setText(
            "---"
        )

        self.output_value.setText(
            "---"
        )

        self.last_played = None

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        print(
            "\nStopping HandNote AI..."
        )

        if self.worker:

            self.worker.stop()

        if self.sound_engine:

            self.sound_engine.cleanup()

        event.accept()


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = HandNoteApp()

    window.show()

    sys.exit(
        app.exec()
    )