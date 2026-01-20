import os
import subprocess
import cv2
from _landmark import landmark


def find_conda_env():
    user_home = os.path.expanduser("~")

    return os.path.join(user_home, "miniconda3", "envs", "ip2p", "python.exe")


def run_pose_edit(image_path, prompt, output_path="../instruct-pix2pix/imgs/output.jpg"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    target_work_dir = os.path.abspath(os.path.join(current_dir, "../instruct-pix2pix"))
    script_path = os.path.join(target_work_dir, "edit_cli.py")

    abs_input_path = os.path.abspath(image_path)
    abs_output_path = os.path.abspath(output_path)

    conda_env = find_conda_env()

    print(target_work_dir)

    cmd = [
        conda_env,
        script_path,
        "--input", abs_input_path,
        "--output", abs_output_path,
        "--edit", prompt,
    ]

    subprocess.run(cmd, cwd=target_work_dir, check=True)

    frame = cv2.imread(abs_output_path)

    ret = landmark().get_landmark(frame)

    print(ret)

    return ret


# if __name__ == "__main__":
#     result = run_pose_edit("test.jpg", "open the arms wider")