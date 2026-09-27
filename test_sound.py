import time

from sound_engine import SoundEngine


print("Starting sound test...")

sound_engine = SoundEngine()

notes = [
    "C",
    "D",
    "E",
    "F",
    "G",
    "A",
    "B"
]

for note in notes:

    print("Playing:", note)

    sound_engine.play_note(note)

    time.sleep(0.8)


sound_engine.cleanup()

print("Sound test completed!")