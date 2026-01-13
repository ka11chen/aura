import json
import os
import random

def create_mock_reference(judge_name, num_frames=30, behavior_type="neutral"):
    """
    Generates synthetic landmark data for a specific judge's characteristic behavior.
    Shape: List[Frames], where Frame = List[Landmarks]
    Landmark = {x, y, z, visibility}
    """
    
    # 33 landmarks for MediaPipe Pose
    # We will just randomize them slightly around a "base pose"
    
    frames = []
    
    for _ in range(num_frames):
        landmarks = []
        for i in range(33):
            # Base position (normalized 0-1)
            base_x = 0.5
            base_y = 0.5
            
            # Modify specific landmarks based on behavior
            
            # Steve Jobs: Steeple (Hands close together, center chest)
            # Left Index(19), Right Index(20) close.
            if behavior_type == "steeple":
                if i == 19: # L Index
                    base_x = 0.48
                    base_y = 0.4
                elif i == 20: # R Index
                    base_x = 0.52
                    base_y = 0.4
                elif i in [15, 17, 21]: # L Wrist, Pinky, Thumb
                    base_x = 0.45 
                    base_y = 0.45
                elif i in [16, 18, 22]: # R Wrist, Pinky, Thumb
                    base_x = 0.55
                    base_y = 0.45
            
            # Trump: Accordion (Hands apart, wide)
            elif behavior_type == "accordion":
                if i == 19: # L Index
                    base_x = 0.3
                    base_y = 0.5
                elif i == 20: # R Index
                    base_x = 0.7
                    base_y = 0.5
                elif i in [15, 17, 21]: # Left Hand
                    base_x = 0.25
                elif i in [16, 18, 22]: # Right Hand
                    base_x = 0.75

            # Add small noise to simulate life/movement
            jitter = 0.01
            
            lm = {
                "x": base_x + random.uniform(-jitter, jitter),
                "y": base_y + random.uniform(-jitter, jitter),
                "z": 0.0,
                "visibility": 0.95
            }
            landmarks.append(lm)
        
        frames.append(landmarks)
        
    return frames

def save_reference_file(filename, data):
    target_dir = os.path.join(os.getcwd(), "reference")
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    print(f"Generated: {filepath}")

if __name__ == "__main__":
    print("Generating Gold Standard Reference Data...")
    
    # Steve Jobs Data (Steeple)
    jobs_data_1 = create_mock_reference("Judge_Steve_Jobs", behavior_type="steeple")
    save_reference_file("Judge_Steve_Jobs_01.json", jobs_data_1)
    
    jobs_data_2 = create_mock_reference("Judge_Steve_Jobs", behavior_type="steeple")
    save_reference_file("Judge_Steve_Jobs_02.json", jobs_data_2)
    
    # Donald Trump Data (Accordion)
    trump_data_1 = create_mock_reference("Judge_Donald_Trump", behavior_type="accordion")
    save_reference_file("Judge_Donald_Trump_01.json", trump_data_1)
    
    trump_data_2 = create_mock_reference("Judge_Donald_Trump", behavior_type="accordion")
    save_reference_file("Judge_Donald_Trump_02.json", trump_data_2)
    
    print("Done.")
