# Eye Blink Counter Application

A real-time eye blink detection application using MediaPipe Face Mesh and OpenCV.

## Features

- Real-time eye blink detection using MediaPipe Face Mesh
- Live video feed with face detection
- Real-time JSON output for each frame
- Optimized for performance with efficient face detection
- Automatic GPU utilization when available
- Lightweight and efficient face detection model

## Requirements

- Python 3.8 or higher
- Webcam
- GPU (optional, will use CPU if not available)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/wellnessatwork/eye-blink-tracker.git
cd eye-blink-tracker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python app.py
```

2. The application will:
   - Open your webcam
   - Display the video feed
   - Track your eye blinks in real-time
   - Output JSON data for each frame containing:
     - Timestamp
     - Total blinks
     - Current FPS
     - Eye state (open/closed)
     - EAR (Eye Aspect Ratio) value
     - Face detection status

---

### Run the JSON blink counter for integration

If you want a minimal, headless script that outputs only the total blink count and EAR in JSON format (for integration with other applications, e.g., Swift/macOS):

```bash
python eye_blink_counter_json.py
```

- This script will output lines like:
  ```json
  {"blinks": 3, "ear": 0.21}
  ```
- The `ear` value is always floored to 2 decimal places.
- Each line is a valid JSON object, suitable for real-time parsing by another application.

---

## Features Description

### Blink Detection
- Uses MediaPipe Face Mesh for accurate facial landmark detection
- Calculates Eye Aspect Ratio (EAR) for precise blink detection
- Real-time visualization of eye landmarks
- Outputs detailed JSON data for each frame

### Performance
- Efficient face detection using MediaPipe
- Automatic GPU utilization when available
- Smooth real-time video processing at 30+ FPS
- Optimized for low CPU usage

## Project Structure

```
eye-blink-tracker/
├── app.py                   # Main application entry point
├── eye_blink_counter_json.py # Minimal JSON output for integration
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
└── README.md               # This documentation
```

## GitHub Repository Setup

To set up this project on your own GitHub:

1. Create a new repository on GitHub
2. Initialize your local repository:
```bash
git init
git add .
git commit -m "Initial commit"
```
3. Link to your GitHub repository:
```bash
git remote add origin https://github.com/yourusername/eye-blink-tracker.git
git push -u origin main
```

## Troubleshooting

If you encounter any issues:

1. Ensure your webcam is properly connected and accessible
2. Check if all dependencies are correctly installed
3. Verify that you have sufficient lighting for face detection
4. Make sure you have the required permissions for webcam access

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 