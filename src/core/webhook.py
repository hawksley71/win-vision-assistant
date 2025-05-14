# webhook.py
# ---------------------------------------------
# This Flask app exposes a webhook endpoint for Home Assistant to trigger the Vision-Aware Smart Assistant.
#
# When to use:
# - Use this file if you want Home Assistant to be able to trigger your assistant (e.g., via automations or voice commands to Google Home/Alexa).
# - This is useful for setups where Home Assistant needs to "push" events or commands to the assistant, such as starting detection or sending custom messages.
#
# When NOT to use:
# - If your assistant only needs to send TTS messages to Home Assistant (to play on a smart speaker), you do NOT need this webhook.
# - In the current Windows setup, the assistant sends TTS to Home Assistant, so webhook.py is not required.
#
# Note:
# - This file is not integrated by default. Enable and configure it only if you want Home Assistant to trigger your assistant via HTTP POST.
# - See project documentation for more details.

from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/trigger", methods=["POST"])
def trigger_assistant():
    print("Trigger received from Home Assistant.")
    try:
        # Adjust the command below to match your assistant's entry point
        subprocess.Popen(["python3", "src/main_assistant.py"])
        return "Assistant triggered.", 200
    except Exception as e:
        return f"Error triggering assistant: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
