"""
UPDATED FRONTEND APP - With Network File Sending
Replace the /api/process-txt route in your original app with this version.

This sends files to the backend receiver PC on your network.
"""

import os
import requests
from flask import Flask, request, redirect, url_for, flash, jsonify

# ====== CONFIGURATION ======
# 🔴 IMPORTANT: Set the backend PC IP address and port
BACKEND_API_URL = (
    "http://192.168.1.100:5001/api/receive-txt"  # Change this to your backend PC IP
)
BACKEND_HEALTH_CHECK = "http://192.168.1.100:5001/api/health"  # Health check endpoint

# You can also detect the backend PC if it's on the same network
# Or use environment variables
import os

BACKEND_IP = os.getenv("BACKEND_IP", "192.168.1.100")
BACKEND_PORT = os.getenv("BACKEND_PORT", "5001")
BACKEND_API_URL = f"http://{BACKEND_IP}:{BACKEND_PORT}/api/receive-txt"
BACKEND_HEALTH_CHECK = f"http://{BACKEND_IP}:{BACKEND_PORT}/api/health"

print(f"Frontend configured to send files to: {BACKEND_API_URL}")

# ====== ROUTE ======


@app.route("/api/process-txt", methods=["POST"])
def process_txt_api():
    """
    Updated route that sends the TXT file to the backend PC on the network
    instead of processing it locally.
    """

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "" or not file.filename.endswith(".txt"):
        return (
            jsonify(
                {"success": False, "message": "Invalid file name or type. Must be .txt"}
            ),
            400,
        )

    try:
        # Check if backend is reachable
        print(f"Checking backend health at {BACKEND_HEALTH_CHECK}...")
        health_response = requests.get(BACKEND_HEALTH_CHECK, timeout=5)

        if health_response.status_code != 200:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Backend is not responding properly (Status: {health_response.status_code})",
                    }
                ),
                503,
            )

        print("✅ Backend is reachable")

    except requests.exceptions.ConnectionError:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Cannot connect to backend at {BACKEND_IP}:{BACKEND_PORT}. Make sure the backend PC is running and the IP is correct.",
                }
            ),
            503,
        )
    except requests.exceptions.Timeout:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Backend health check timed out. Check network connection.",
                }
            ),
            503,
        )
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error checking backend: {str(e)}"}),
            503,
        )

    try:
        # Prepare the file for sending
        print(f"Sending file '{file.filename}' to backend...")

        files = {"file": (file.filename, file.stream, "text/plain")}

        # Send the file to the backend
        response = requests.post(
            BACKEND_API_URL, files=files, timeout=60  # 60 second timeout for processing
        )

        result = response.json()

        if response.status_code == 200 and result.get("success"):
            print(f"✅ File processed successfully by backend")
            print(f"Output: {result.get('output', '')[:100]}...")

            return (
                jsonify(
                    {
                        "success": True,
                        "message": "File uploaded and processed by backend",
                        "backend_response": result,
                    }
                ),
                200,
            )
        else:
            error_msg = result.get("message", "Unknown error from backend")
            print(f"❌ Backend error: {error_msg}")

            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Backend processing failed: {error_msg}",
                    }
                ),
                500,
            )

    except requests.exceptions.Timeout:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Request to backend timed out. Processing may be taking too long.",
                }
            ),
            504,
        )
    except requests.exceptions.ConnectionError as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Connection error with backend: {str(e)}",
                }
            ),
            503,
        )
    except Exception as e:
        print(f"❌ Error sending file to backend: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error communicating with backend: {str(e)}",
                }
            ),
            500,
        )


# ====== HELPER ROUTE ======


@app.route("/api/backend-status", methods=["GET"])
def backend_status():
    """Check if the backend PC is online."""
    try:
        response = requests.get(BACKEND_HEALTH_CHECK, timeout=5)
        if response.status_code == 200:
            return (
                jsonify({"status": "online", "message": "Backend is online and ready"}),
                200,
            )
    except:
        pass

    return (
        jsonify(
            {
                "status": "offline",
                "message": f"Backend at {BACKEND_IP}:{BACKEND_PORT} is not responding",
            }
        ),
        503,
    )
