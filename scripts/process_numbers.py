# scripts/process_numbers.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tqdm import tqdm

print("🔢 PROCESSING NUMBER DATASET (0-9 ONLY)")

class NumberProcessor:
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
    
    def process_numbers(self):
        """Process only 0-9 numbers"""
        print("📁 Looking for number dataset...")
        base_path = "asl_number_dataset"
        
        if not os.path.exists(base_path):
            print("❌ Number dataset not found!")
            return False
            
        # Only 0-9
        numbers = [str(i) for i in range(10)]  # 0 to 9
        
        total_success = 0
        for number in numbers:
            class_path = os.path.join(base_path, number)
            if not os.path.exists(class_path):
                print(f"❌ Number folder {number} not found!")
                continue
                
            images = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]
            # Use all 70 images per number
            successful = 0
            for img_name in tqdm(images, desc=f"Number {number}", leave=False):
                img_path = os.path.join(class_path, img_name)
                landmarks = self.extract_landmarks(img_path)
                
                if landmarks is not None:
                    self.landmarks_data.append(landmarks)
                    self.labels.append(number)
                    successful += 1
            
            total_success += successful
            print(f"   ✅ {number}: {successful}/{len(images)} images processed")
            
        print(f"🎉 NUMBER PROCESSING COMPLETE: {total_success} images")
        return True
    
    def save_number_data(self):
        """Save processed number data"""
        if len(self.landmarks_data) == 0:
            print("❌ No data to save!")
            return False
            
        data = {
            'landmarks': np.array(self.landmarks_data),
            'labels': np.array(self.labels)
        }
        
        # Save to datasets folder
        with open('datasets/numbers/number_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Number data saved: {data['landmarks'].shape}")
        return True
    
    def close(self):
        self.hands.close()

# RUN THE PROCESSOR
if __name__ == "__main__":
    processor = NumberProcessor()
    
    try:
        print("=" * 50)
        print("STARTING NUMBER DATA PROCESSING")
        print("=" * 50)
        
        success = processor.process_numbers()
        
        if success:
            processor.save_number_data()
            print("\n🎉 NUMBER DATA READY FOR TRAINING!")
            print("Next: Run train_numbers.py")
        else:
            print("❌ Failed to process number data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.close()