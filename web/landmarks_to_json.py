import json
import os

def save_landmarks_to_file(result_list, filename="landmarks.json", is_reference=False):
    base_dir = ".coding"
    simplified_data = []

    def parse_node(p):
        return {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "visibility": getattr(p, 'visibility', 0.0)
        }

    if is_reference:
        target_dir = os.path.join(base_dir, "reference")
    else:
        target_dir = base_dir
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)

    if result_list and result_list.pose_landmarks and len(result_list.pose_landmarks) > 0:
        for p in result_list.pose_landmarks[0]:
            simplified_data.append(parse_node(p))

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(simplified_data, f)

    print(f"Data saved to: {file_path}")