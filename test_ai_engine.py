from ai_engine import AIEngine


print("=" * 60)
print("Testing HandNote AI Engine")
print("=" * 60)


engine = AIEngine()


print("\nCamera started.")
print("Press Q to quit.")
print("Press R to reset.")


while True:

    result = engine.get_frame()

    if result is None:

        print("Could not read camera frame.")
        break


    frame, data = result


    print(
        f"\r"
        f"Left: {data['left']} "
        f"({data['left_confidence'] * 100:.1f}%) | "
        f"Right: {data['right']} "
        f"({data['right_confidence'] * 100:.1f}%) | "
        f"Output: {data['output']}",
        end=""
    )


    import cv2

    cv2.imshow(
        "AI Engine Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("r"):

        engine.reset()

        print("\nGesture reset.")


engine.release()

cv2.destroyAllWindows()

print("\n\nTest completed.")