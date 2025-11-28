import subprocess
from flask import Flask, request

app = Flask(__name__)

TMUX_SESSION = "chill"  # change to your tmux session name


@app.route("/receive", methods=["POST"])
def receive():
    if "file" not in request.files:
        return "No file found in request", 400

    request.files["file"].read()

    print("\n--- OPENING NEW TMUX PANE & RUNNING hello.py ---")

    # Run tmux via shell
    cmd = f'tmux split-window -t {TMUX_SESSION} -v "python3 hello.py"'
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("--------------------------------\n")

    return "hello.py started in new tmux pane", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
