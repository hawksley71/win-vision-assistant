# Vision-Aware Smart Assistant

## Project Overview
This project is a modular, voice-driven, vision-aware assistant that integrates live object detection (YOLOv8), historical log analysis, and natural language voice interaction. The system can answer both live and historical queries about detected objects, using the current camera feed and combined detection logs. It features robust intent detection (OpenAI + regex), partial/fuzzy matching, and seamless integration with Home Assistant for text-to-speech (TTS) output. If the smart speaker is unavailable, it falls back to local TTS playback.

## Technologies Used
- **Python 3.10**
- **YOLOv8** for object detection (GPU-accelerated with CUDA when available)
- **OpenAI API** for intent detection and pattern analysis
- **SpeechRecognition** and **gTTS** for voice input/output
- **OpenCV** for camera and image processing
- **Home Assistant** for TTS and smart home integration
- **Pandas, NumPy, scikit-learn** for data analysis
- **Requests, python-dotenv** for API and environment management

## Home Assistant Integration
Home Assistant is used to play TTS responses on a smart speaker. The assistant sends HTTP requests to Home Assistant's TTS service, which vocalizes responses. If the TTS service fails, the system uses local TTS playback via gTTS and PyGame.

## Optional: Home Assistant Webhook Integration
- The file `src/core/webhook.py` provides a Flask webhook endpoint for Home Assistant to trigger the Vision-Aware Smart Assistant (e.g., via automations or voice commands).
- **When to use:** If you want Home Assistant to "push" events or commands to your assistant (such as starting detection or sending custom messages).
- **Not needed:** If your assistant only sends TTS messages to Home Assistant (the default/current setup), you do NOT need to run or configure this webhook.
- See comments in `webhook.py` for more details.

## Setup Instructions

### Linux & Windows Setup
1. Install [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) or [Conda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:
   ```
   mamba env create -f docs/environment.yml
   # or
   conda env create -f docs/environment.yml
   ```
3. Activate the environment:
   ```
   mamba activate vision-assistant
   # or
   conda activate vision-assistant
   ```
4. **Install PyTorch with CUDA (Windows):**
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
5. Set up your `.env` file with OpenAI and Home Assistant tokens.
6. Download YOLOv8 weights (`yolov8n.pt`) and place in `models/weights/`.
7. Run the main assistant script:
   ```
   python -m src.core.voice_loop
   ```

### Home Assistant Setup (Windows)
- Install Docker Desktop.
- Run Home Assistant container:
  ```
  docker run -d --name homeassistant --privileged -v C:\Users\hawks\homeassistant:/config -e TZ=America/New_York --network=host homeassistant/home-assistant:stable
  ```
- Complete onboarding at [http://localhost:8123](http://localhost:8123).
- Add Google Cast integration for smart speaker support.

## Codebase Structure
- `src/core/assistant.py`: Main detection and voice assistant logic.
- `src/core/voice_loop.py`: Manages the main loop for voice and detection.
- `src/models/yolov8_model.py`: YOLOv8 model wrapper.
- `src/utils/audio.py`: Microphone selection utilities.
- `src/config/settings.py`: Centralized configuration.
- `data/raw/`, `data/processed/`: Detection logs.
- `models/weights/`: Model weights (tracked in Git).

## Troubleshooting & Performance Tips
- Ensure PyTorch is GPU-enabled: `python -c "import torch; print(torch.cuda.is_available())"`
- Lower camera resolution in `src/config/settings.py` for higher FPS.
- Use Docker for Home Assistant on Windows.
- If TTS fails, check Home Assistant logs and network settings.

## Demo Script
- See `docs/assistant_test_questions.txt` for example queries.

## Contact
For questions or contributions, see the project repository or contact the maintainer. 