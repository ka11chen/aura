import os
import subprocess
import cv2
from _landmark import landmark


def find_conda_env():
    user_home = os.path.expanduser("~")

    return os.path.join(user_home, "miniconda3", "envs", "ip2p", "python.exe")


def run_pose_edit(image_path, advice, output_path="../instruct-pix2pix/imgs/output.jpg"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    target_work_dir = os.path.abspath(os.path.join(current_dir, "../instruct-pix2pix"))
    script_path = os.path.join(target_work_dir, "edit_cli.py")

    abs_input_path = os.path.abspath(image_path)
    abs_output_path = os.path.abspath(output_path)

    conda_env = find_conda_env()

    print(target_work_dir)

    image_guidance = 0.15
    text_guidance = 10.0
    steps = 20

    prompt = f"Make a single person stand facing the camera, giving a TED talk presentation, {advice}"

    cmd = [
        conda_env,
        script_path,
        "--input", abs_input_path,
        "--output", abs_output_path,
        "--edit", prompt,
        "--cfg-image", str(image_guidance),
        "--cfg-text", str(text_guidance),
        "--steps", str(steps)
    ]

    subprocess.run(cmd, cwd=target_work_dir, check=True)

    frame = cv2.imread(abs_output_path)

    ret = landmark().get_landmark(frame)

    print(ret)

    return ret

if __name__ == "__main__":
    result = run_pose_edit("test.jpg", "open the arms wider")