# scripts/process_letters.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tqdm import tqdm

print("🔤 PROCESSING ALPHABET DATASET (A-Z ONLY)")

class LetterProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.landmarks_data = []
        self.labels = []
        
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from an image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for landmark in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            return None
        except Exception as e:
            return None
    
    def process_letters(self):
        """Process only A-Z letters"""
        print("📁 Looking for alphabet dataset...")
        base_path = "asl_alphabet_dataset/asl_alphabet_train"
        
        if not os.path.exists(base_path):
            print("❌ Alphabet dataset not found!")
            return False
            
        # Only A-Z (no del, nothing, space)
        letters = [chr(i) for i in range(65, 91)]  # A to Z
        
        total_success = 0
        for letter in letters:
            class_path = os.path.join(base_path, letter)
            if not os.path.exists(class_path):
                print(f"❌ Letter folder {letter} not found!")
                continue
                
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
            # Use 300 images per letter
            images = images[:300]
            
            successful = 0
            for img_name in tqdm(images, desc=f"Letter {letter}", leave=False):
                img_path = os.path.join(class_path, img_name)
                landmarks = self.extract_landmarks(img_path)
                
                if landmarks is not None:
                    self.landmarks_data.append(landmarks)
                    self.labels.append(letter)
                    successful += 1
            
            total_success += successful
            print(f"   ✅ {letter}: {successful}/{len(images)} images processed")
            
        print(f"🎉 ALPHABET PROCESSING COMPLETE: {total_success} images")
        return True
    
    def save_letter_data(self):
        """Save processed letter data"""
        if len(self.landmarks_data) == 0:
            print("❌ No data to save!")
            return False
            
        data = {
            'landmarks': np.array(self.landmarks_data),
            'labels': np.array(self.labels)
        }
        
        # Save to datasets folder
        with open('datasets/letters/letter_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Letter data saved: {data['landmarks'].shape}")
        return True
    
    def close(self):
        self.hands.close()

# RUN THE PROCESSOR
if __name__ == "__main__":
    processor = LetterProcessor()
    
    try:
        print("=" * 50)
        print("STARTING ALPHABET DATA PROCESSING")
        print("=" * 50)
        
        success = processor.process_letters()
        
        if success:
            processor.save_letter_data()
            print("\n🎉 ALPHABET DATA READY FOR TRAINING!")
            print("Next: Run train_letters.py")
        else:
            print("❌ Failed to process alphabet data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.close()