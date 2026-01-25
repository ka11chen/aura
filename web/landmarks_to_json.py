import json
import os

def save_landmarks_to_file(result_list, filename="landmarks.json", is_reference=False):
    base_dir = ".coding"

    if is_reference:
        target_dir = os.path.join(base_dir, "reference")
    else:
        target_dir = base_dir

    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, filename)

    simplified_data = []
    for result in result_list:
        if result and hasattr(result, 'pose_landmarks'):
            for frame in result.pose_landmarks:
                frame_points = []
                for p in frame:
                    frame_points.append({
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "visibility": getattr(p, 'visibility', 0.0)
                    })
                simplified_data.append(frame_points)

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(simplified_data, f)

    print(f"Data saved to: {file_path}")