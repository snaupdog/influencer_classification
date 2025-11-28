import subprocess
from flask import Flask, request

app = Flask(__name__)

TMUX_SESSION = "chill"  # <- your tmux session name


@app.route("/receive", methods=["POST"])
def receive():
    if "file" not in request.files:
        return "No file found in request", 400

    file = request.files["file"]
    _ = file.read()

    print("\n--- RUNNING PIPELINE IN TMUX ---")

    # Pipeline command
    # pipeline_cmd = "python3 run_pipeline.py --influencers 10 --posts 2"
    # pipeline_cmd = "python3 run_pipeline.py --influencers 10 --posts 5"
    pipeline_cmd = "python3 run_pipeline.py --influencers 500 --posts 10"

    # The pipeline needs 3 y confirmations
    auto_input = 'printf "y\ny\ny\n" | ' + pipeline_cmd + "; exec bash"

    # ESCAPE everything properly for tmux
    tmux_cmd = (
        f"tmux split-window -t {TMUX_SESSION} -v "
        f'"printf \\"y\\\\ny\\\\ny\\\\n\\" | {pipeline_cmd}; exec bash"'
    )

    result = subprocess.run(["bash", "-c", tmux_cmd], capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("--------------------------------\n")

    return "Pipeline executed in tmux", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
