import cv2
import mediapipe as mp

# Initialize mediapipe drawing and hands solution
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Capture video from webcam
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Flip the image horizontally and convert color to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(image_rgb)

    # Convert the image back to BGR for OpenCV display
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)
            )

    # Show the result in a window
    cv2.imshow('Hand Tracker', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
