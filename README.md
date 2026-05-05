# Smart Attendance System using Face Recognition

A production-ready, CPU-only attendance system that registers users through a webcam, builds Dlib/`face_recognition` embeddings, recognizes faces in real time, prevents duplicate attendance within a cooldown window, and exposes a Streamlit dashboard plus CLI and optional FastAPI entry points.

## Folder Structure

```text
smart_attendance_system/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── .gitkeep
├── encodings/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── api.py
    ├── attendance.py
    ├── cloud_sync.py
    ├── config.py
    ├── encode.py
    ├── liveness.py
    ├── recognize.py
    ├── register.py
    ├── utils.py
    └── vector_index.py
```

## What It Includes

- Webcam registration with 20-30 images per user.
- Face validation that rejects frames with no face, multiple faces, poor lighting, or blur.
- Dataset layout: `dataset/{person_name}/image.jpg`.
- Dlib HOG detection and ResNet 128-dimensional face embeddings through `face_recognition`.
- Encoding cache at `encodings/known_faces.pkl`; rebuilds only when dataset files change.
- Real-time CPU recognition with 50% downscaling and alternate-frame processing.
- Euclidean distance matching with configurable threshold, default `0.55`.
- Confidence classes:
  - `< 0.45`: `Valid`
  - `0.45-0.55`: `Probable`
  - `>= 0.55`: `Invalid`
- CSV attendance by default at `logs/attendance.csv`.
- SQLite backend at `logs/attendance.db`.
- 60-minute duplicate prevention using in-memory state plus log/database history.
- Unknown face display, bounding boxes, name labels, confidence, timestamp overlay, and clean camera release.
- Optional blink-based liveness check.
- Optional FastAPI endpoints and cloud-sync provider hooks.

## Setup

Use Python 3.9 or newer. Python 3.10 or 3.11 is usually the smoothest choice for `dlib`.

### macOS

```bash
cd smart_attendance_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `dlib` fails to build, install system tools first:

```bash
xcode-select --install
brew install cmake
pip install -r requirements.txt
```

### Windows

Install Python 3.10 or 3.11, CMake, and Visual Studio Build Tools with C++ support, then run:

```powershell
cd smart_attendance_system
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Linux

```bash
cd smart_attendance_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
pip install -r requirements.txt
```

## How to Run

### Streamlit Dashboard

```bash
cd smart_attendance_system
source .venv/bin/activate
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Dashboard sections:

- `Dashboard`: metrics, recent logs, encoding cache state.
- `Register User`: webcam sample capture.
- `Live Attendance`: real-time recognition and automatic attendance marking.
- `Logs`: view and export attendance records.
- `Dataset Health`: validate dataset and rebuild encodings.
- `Setup`: common commands.

### CLI Workflow

Register a user:

```bash
python -m src.register "Priya Sharma" --samples 25
```

Build or rebuild encodings:

```bash
python -m src.encode --force --validate
```

Start attendance with CSV logging:

```bash
python -m src.recognize --storage csv
```

Start attendance with SQLite logging:

```bash
python -m src.recognize --storage sqlite
```

Require a blink before marking:

```bash
python -m src.recognize --liveness
```

Press `q` in the OpenCV window to exit cleanly.

### Optional API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /attendance?limit=100&storage=csv`
- `POST /encodings/rebuild`

## Sample Dataset Instructions

Recommended capture rules for each person:

1. Capture 20-30 samples.
2. Use one person per frame.
3. Include normal indoor lighting, slightly brighter light, and slightly dimmer light.
4. Include front, slight-left, slight-right, chin-up, and chin-down angles.
5. Avoid masks, heavy sunglasses, motion blur, and strong backlight.

Manual dataset placement is also supported:

```text
dataset/
├── Priya_Sharma/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── Arjun_Mehta/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

After adding images manually, rebuild encodings:

```bash
python -m src.encode --force --deep --validate
```

## Attendance Output

CSV and SQLite rows use this schema:

```text
name,date,time,timestamp,confidence,distance,status,source
Priya_Sharma,2026-05-05,09:31:04,2026-05-05T09:31:04,62.18,0.3782,Valid,camera
```

Duplicate prevention checks the most recent mark for each person and blocks new entries for 60 minutes. The cooldown can be changed in `src/config.py`.

## Performance Notes

Default CPU settings are tuned for standard laptops:

- HOG detector, not CNN.
- Frame scale: `0.5`.
- Process every second frame.
- Dlib encoding model: `small`.
- NumPy L2 search by default.
- Optional FAISS hook in `src/vector_index.py` for larger deployments.

Expected performance depends on CPU and webcam resolution, but the defaults target roughly 15 FPS on common laptop CPUs.

## Error Handling

Handled cases:

- No webcam or wrong camera index.
- Frame read failure.
- No face detected during registration.
- Multiple faces during registration.
- Dark, overexposed, or blurry registration frames.
- Corrupted dataset images.
- Missing or corrupted encoding cache.
- Unknown faces during recognition.
- Duplicate attendance attempts within cooldown.

System logs are written to `logs/system.log`.

## Example Output Screenshots

Screenshot descriptions:

- Registration screen: webcam feed with a face bounding box, sample counter like `Priya_Sharma: 14/25`, and pose prompts such as `Turn slightly left`.
- Attendance screen: live video with green `Valid` boxes, amber `Probable` boxes, red `Unknown` boxes, confidence percentages, and timestamp overlay.
- Logs screen: sortable attendance table with an `Export CSV` button.
- Dataset Health screen: per-person image counts and deep validation errors for problematic images.

## Deployment Checklist

1. Create a dedicated machine account for the operator.
2. Run registration in controlled lighting.
3. Keep `dataset/`, `encodings/`, and `logs/` outside public git repositories.
4. Rebuild encodings after every registration batch.
5. Test unknown-person behavior before going live.
6. Use SQLite for multi-day institutional records.
7. Back up `logs/attendance.csv` or `logs/attendance.db`.
8. Implement a real provider from `src/cloud_sync.py` for cloud backup.
9. Place the app behind authenticated access if Streamlit is hosted on a network.

