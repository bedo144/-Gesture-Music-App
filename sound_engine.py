import pygame
import numpy as np


class SoundEngine:

    def __init__(self):

        pygame.mixer.init(
            frequency=44100,
            size=-16,
            channels=2,
            buffer=512
        )

        self.sample_rate = 44100

        self.note_frequencies = {

            "C": 261.63,
            "C#": 277.18,

            "D": 293.66,
            "D#": 311.13,

            "E": 329.63,

            "F": 349.23,
            "F#": 369.99,

            "G": 392.00,
            "G#": 415.30,

            "A": 440.00,
            "A#": 466.16,

            "B": 493.88
        }

        self.sounds = {}

        self._generate_sounds()

    # ==========================================================
    # GENERATE NOTES
    # ==========================================================

    def _generate_sounds(self):

        print("Generating piano sounds...")

        for note, frequency in self.note_frequencies.items():

            self.sounds[note] = self._create_note(
                frequency,
                duration=0.7
            )

        print("Sound engine ready!")

    # ==========================================================
    # CREATE NOTE
    # ==========================================================

    def _create_note(self, frequency, duration=0.7):

        t = np.linspace(
            0,
            duration,
            int(self.sample_rate * duration),
            False
        )

        wave_data = (

            1.00 * np.sin(
                2 * np.pi * frequency * t
            )

            + 0.50 * np.sin(
                2 * np.pi * frequency * 2 * t
            )

            + 0.25 * np.sin(
                2 * np.pi * frequency * 3 * t
            )

            + 0.12 * np.sin(
                2 * np.pi * frequency * 4 * t
            )
        )

        envelope = np.exp(
            -3.5 * t
        )

        wave_data *= envelope

        max_value = np.max(
            np.abs(wave_data)
        )

        if max_value > 0:

            wave_data /= max_value

        audio = np.int16(
            wave_data * 32767
        )

        audio = np.column_stack(
            (
                audio,
                audio
            )
        )

        sound = pygame.sndarray.make_sound(
            audio
        )

        return sound

    # ==========================================================
    # PLAY SINGLE NOTE
    # ==========================================================

    def play_note(self, note):

        if note not in self.sounds:

            print(
                f"Warning: Note '{note}' does not exist."
            )

            return

        self.sounds[note].play()

    # ==========================================================
    # CHORD DEFINITIONS
    # ==========================================================

    def get_chord(self, root, chord_type):

        major_chords = {

            "C":  ["C", "E", "G"],
            "D":  ["D", "F#", "A"],
            "E":  ["E", "G#", "B"],
            "F":  ["F", "A", "C"],
            "G":  ["G", "B", "D"],
            "A":  ["A", "C#", "E"],
            "B":  ["B", "D#", "F#"]
        }

        minor_chords = {

            "C":  ["C", "D#", "G"],
            "D":  ["D", "F", "A"],
            "E":  ["E", "G", "B"],
            "F":  ["F", "G#", "C"],
            "G":  ["G", "A#", "D"],
            "A":  ["A", "C", "E"],
            "B":  ["B", "D", "F#"]
        }

        if chord_type == "Major":

            return major_chords.get(
                root,
                []
            )

        elif chord_type == "Minor":

            return minor_chords.get(
                root,
                []
            )

        return []

    # ==========================================================
    # PLAY CHORD
    # ==========================================================

    def play_chord(self, root, chord_type):

        chord = self.get_chord(
            root,
            chord_type
        )

        if not chord:

            print(
                f"Warning: No chord found for "
                f"{root} {chord_type}"
            )

            return

        print(
            f"🎹 Playing {root} {chord_type}: "
            f"{' + '.join(chord)}"
        )

        # Play all notes simultaneously
        for note in chord:

            if note in self.sounds:

                self.sounds[note].play()

    # ==========================================================
    # STOP ALL SOUND
    # ==========================================================

    def stop(self):

        pygame.mixer.stop()

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def cleanup(self):

        pygame.mixer.quit()