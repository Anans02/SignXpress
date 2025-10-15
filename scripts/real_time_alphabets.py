# scripts/real_time_alphabets.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import Counter

print("🔤 REAL-TIME ASL ALPHABET DETECTOR")
print("Loading alphabet model...")

# Load alphabet model
try:
    model = keras.models.load_model('models/alphabet_model.h5')
    with open('models/letter_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("✅ Alphabet model loaded!")
    print(f"🎯 Can detect: A to Z ({len(label_encoder.classes_)} letters)")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n🚀 READY! Show ASL letters (A-Z) to the camera!")
print("📝 Instructions:")
print("   - Show ONE hand gesture at a time")
print("   - Keep your hand clearly visible")
print("   - Press 'Q' to quit")
print("   - Press 'C' to clear current prediction")

# Variables for smooth predictions
prediction_history = []
current_letter = "Show Hand"
confidence = 0.0
target_letter = "A"  # You can change this for your lessons

while True:
    # Read camera frame
    success, frame = cap.read()
    if not success:
        break
    
    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # If hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict the letter
            predictions = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            current_confidence = np.max(predictions[0])
            predicted_letter = label_encoder.inverse_transform([predicted_idx])[0]
            
            # Smooth the confidence
            confidence = 0.7 * confidence + 0.3 * current_confidence
            
            # Add to prediction history for stability
            if current_confidence > 0.6:
                prediction_history.append(predicted_letter)
                if len(prediction_history) > 5:
                    prediction_history.pop(0)
            
            # Get most common prediction from history
            if len(prediction_history) >= 3:
                most_common = Counter(prediction_history).most_common(1)[0]
                current_letter = most_common[0]
            
            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Choose color based on confidence
            if confidence > 0.85:
                color = (0, 255, 0)  # Green - High confidence
                status = "EXCELLENT"
            elif confidence > 0.70:
                color = (0, 255, 255)  # Yellow - Good confidence
                status = "GOOD"
            elif confidence > 0.55:
                color = (0, 165, 255)  # Orange - Medium confidence
                status = "OK"
            else:
                color = (0, 0, 255)  # Red - Low confidence
                status = "LOW"
            
            # Display prediction information
            cv2.putText(frame, f"TARGET: {target_letter}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(frame, f"YOUR SIGN: {current_letter}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, f"Status: {status}", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show "GOOD JOB!" if correct with high confidence
            if current_letter == target_letter and confidence > 0.8:
                cv2.putText(frame, "GOOD JOB! ✅", (20, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    else:
        # No hand detected
        prediction_history = []
        confidence = 0.0
        cv2.putText(frame, "SHOW YOUR HAND 👋", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Make sure hand is clearly visible", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Instructions at bottom
    cv2.putText(frame, "Show ASL letters: A to Z", (20, frame.shape[0] - 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to quit | 'C' to clear", (20, frame.shape[0] - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('ASL Alphabet Teacher - SignXpress', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        prediction_history = []
        current_letter = "Show Hand"
        print("🔄 Prediction cleared!")

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Alphabet detector closed successfully!")