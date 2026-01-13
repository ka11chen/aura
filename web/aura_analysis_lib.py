import json
import os
import glob
import numpy as np

# Standard Landmark Map for reference (Agents can use this if needed, but usually they key off IDs)
LANDMARK_MAP = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb"
}

def load_json_safely(path):
    """Loads JSON from path, returning None if failed."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_frames(data):
    """
    Normalizes input data into List[List[LandmarkDict]].
    Handles cases where data is {'landmarks': [...]}, List[...], etc.
    """
    if data is None: 
        return []
    
    # If it's a dict with 'landmarks' key (common mistake), extract it
    if isinstance(data, dict) and "landmarks" in data:
        data = data["landmarks"]

    # If it's a list, check if it's a list of frames or list of landmarks (single frame)
    if isinstance(data, list):
        if not data:
            return []
        
        # Check first element
        first = data[0]
        
        # Case 1: List of Frames (List[List[Landmark]]) -> Valid
        if isinstance(first, list):
            return data
            
        # Case 2: Single Frame (List[Landmark]) -> Wrap in list
        if isinstance(first, dict) and 'x' in first:
            return [data]

    print("Warning: Unknown data structure. Returning empty.")
    return []

def load_reference_data(judge_name):
    """
    Loads all reference files for a judge.
    Returns: List[List[List[Landmark]]] (List of clips, each clip is list of frames)
    """
    # Clean name (e.g. "Donald Trump" -> "Donald_Trump")
    safe_name = judge_name.replace(" ", "_")
    
    # Try multiple patterns
    patterns = [
        f"reference/Judge_{safe_name}_*.json",
        f"reference/{safe_name}_*.json",
        f"reference/*{safe_name}*.json"
    ]
    
    files = []
    for p in patterns:
        found = glob.glob(p)
        files.extend(found)
        
    files = list(set(files)) # Unique
    
    if not files:
        print(f"CRITICAL: No reference files found for {judge_name} (patterns: {patterns})")
        return []

    print(f"Loading {len(files)} reference files: {files}")
    
    loaded_clips = []
    for f in files:
        data = load_json_safely(f)
        frames = extract_frames(data)
        if frames:
            loaded_clips.append(frames)
            
    return loaded_clips

def calculate_stats(values):
    """Returns min, max, mean, std."""
    if not values:
        return 0, 0, 0, 0
    vals = np.array(values)
    return float(np.min(vals)), float(np.max(vals)), float(np.mean(vals)), float(np.std(vals))

def evaluate_verdict(user_val, ref_min, ref_max, ref_mean):
    """
    Determines severity and verdict based on range logic.
    Severity 0.0-0.2: Inside range (Good)
    Severity 0.3-0.5: Near edge (Acceptable)
    Severity 0.6-0.8: Outside but close (Warning)
    Severity 0.9-1.0: Far outside (Critical)
    """
    # Simple range check
    range_span = max(ref_max - ref_min, 0.0001) # Avoid div0
    
    if ref_min <= user_val <= ref_max:
        # Inside range. How close to center?
        dist_to_center = abs(user_val - ref_mean)
        max_dist = range_span / 2
        normalized_dev = dist_to_center / max_dist if max_dist > 0 else 0
        # Map 0.0 (center) -> 1.0 (edge) to Severity 0.0 -> 0.3
        return 0.0 + (normalized_dev * 0.3)
    else:
        # Outside range
        dist_to_min = abs(user_val - ref_min)
        dist_to_max = abs(user_val - ref_max)
        dist_to_edge = min(dist_to_min, dist_to_max)
        
        # How many range_spans away?
        factor = dist_to_edge / range_span
        
        if factor < 0.5:
            return 0.4 + (factor * 0.4) # 0.4 - 0.6
        elif factor < 1.0:
            return 0.6 + (factor * 0.2) # 0.6 - 0.8
        else:
            return 0.9 + min((factor - 1.0) * 0.1, 0.1) # 0.9 - 1.0

def run_analysis(judge_name, metric_func):
    """
    Main entry point.
    1. Loads User Data + Reference Data
    2. Runs metric_func(frame) on all frames
    3. Aggregates results (Mean over time)
    4. Compares User Mean vs Reference Stats
    5. Returns JSON string for the Judge to parse.
    """
    print(f"--- Starting Analysis for {judge_name} ---")
    
    # 1. Load User Data
    user_data = load_json_safely("landmarks.json")
    user_frames = extract_frames(user_data)
    
    # 2. Load Reference Data
    ref_clips = load_reference_data(judge_name)
    
    if not user_frames:
        return json.dumps({
            "error": "No User Data",
            "description": "Could not load landmarks.json or it was empty."
        })
        
    if not ref_clips:
        # FAIL SAFE: Should NOT happen if files exist, but if it does, return valid JSON structure so Judge doesn't crash
        return json.dumps({
            "metric_name": "Unknown",
            "user_value": 0,
            "ref_min": 0,
            "ref_max": 0,
            "ref_mean": 0,
            "status": "Reference Data Missing (Check implementation)",
            "severity": 1.0
        })

    # 3. Compute Metric
    # Helper to average over a clip (list of frames)
    def process_clip(frames):
        vals = []
        for frame in frames:
            try:
                val = metric_func(frame)
                if val is not None:
                    vals.append(val)
            except Exception:
                pass
        return np.mean(vals) if vals else None

    # User Score
    user_score = process_clip(user_frames)
    
    # Ref Scores (one score per clip/file)
    ref_scores = []
    for clip in ref_clips:
        s = process_clip(clip)
        if s is not None:
            ref_scores.append(s)
            
    if user_score is None:
        return json.dumps({"error": "Computation Failed", "description": "Metric returned None for all user frames."})
        
    if not ref_scores:
         return json.dumps({"error": "Reference Computation Failed", "description": "Metric returned None for all reference frames."})

    # 4. Stats
    r_min, r_max, r_mean, _ = calculate_stats(ref_scores)
    
    # 5. Verdict
    return json.dumps({
        "metric_name": "Custom Metric",
        "user_value": user_score,
        "ref_min": r_min,
        "ref_max": r_max,
        "ref_mean": r_mean
    })
