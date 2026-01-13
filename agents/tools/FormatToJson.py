import json
import sys

def md_to_json_string(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Convert to a JSON-escaped string
    json_string = json.dumps(content)

    print(json_string)


if __name__ == "__main__":
    # print("Usage: python FormatToJson.py agent.in")
    md_file = sys.argv[1]
    md_to_json_string(md_file)

