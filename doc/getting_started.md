# Welcome to **AURA**

AURA is a tool for learning body language and presentation posture, based on feedback from multiple judges and user preferences.
Suggestions are not generated directly by AI. Instead, each suggestion is computed from specific features associated with individual judges.

This programs use `flask`, `mediapipe` for motion tracking, `autogen` AI agents, and `instruct-pix2pix` for generating modified skeletons.
## Pipeline

### Select reference
- Use default judges.
- (Or) Import your judges by uploading their names and reference images. 

### Record
- Press `開始錄製` button in the center. The program will record the camera for 30 seconds.
    - The program will only value the pose, so speaking content is not important.
- (Or) Upload an MP4 video.

### Suggestions
- View suggestions given by different judges sorted by the severity.
- You will see some frames of the recordings, the extracted skeleton of the images, and a suggested pose that best addresses the most severe problem.

### Feedback
- Use the `Accept` and `Reject` button for each suggestions.
- The feedback will change the weight of that judge, e.g., the suggestions given by that judge will be considered more/less important in future use.

### Retry
- Users are encouraged to use the website multiple times consecutively to find the suitble preference weights and improve their presentation performances. 