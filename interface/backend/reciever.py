# receiver.py
import subprocess
from flask import Flask, request

app = Flask(__name__)


@app.route("/receive", methods=["POST"])
def receive():
    if "file" not in request.files:
        return "No file found in request", 400

    file = request.files["file"]
    _ = file.read()  # you don’t need the text anymore

    # --- Run your python script instead of printing ---
    print("\n--- RUNNING hello.py ---")

    result = subprocess.run(["python3", "hello.py"], capture_output=True, text=True)

    print(result.stdout)
    print("--------------------------------\n")

    return "Python script executed", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
