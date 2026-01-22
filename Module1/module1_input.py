import cv2  # Import the OpenCV library

def start_webcam():
    # 1. Initialize the webcam
    # '0' usually refers to the default built-in webcam.
    # If you have an external USB camera, you might need to change this to 1.
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    # 2. Start the Loop to process video frames continuously
    while True:
        # Read a single frame from the webcam
        # 'ret' is a boolean (True/False) if the frame was read correctly
        # 'frame' is the actual image captured
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # --- SPACE FOR FUTURE MODULE 1 LOGIC ---
        # Later, we will insert the Object Detection & Scene code here.
        # For now, we just pass the frame through.
        # ---------------------------------------

        # 3. Display the resulting frame in a window
        cv2.imshow('Voice Vision - Input Test', frame)

        # 4. Quit logic
        # Wait for 1 millisecond and check if the user pressed 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Cleanup
    # Release the webcam resource and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam()