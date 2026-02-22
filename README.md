# Face Recognition (InsightFace-Based)

A face recognition project built with **Python**, **InsightFace**, and **ONNX Runtime** for detecting and recognizing faces from images/video streams.

This project is a practical foundation for:
- face identification
- attendance / access logging workflows
- real-time camera-based recognition
- experimentation with InsightFace models

---

## Features

- Face detection and recognition using **InsightFace**
- Supports **CPU** (`onnxruntime`) and optional **GPU** (`onnxruntime-gpu`)
- Easy local setup with Python virtual environment
- Modular structure for extending:
  - image-based recognition
  - webcam/live stream recognition
  - database / attendance logging integration
- Windows-friendly installation notes (including build-tools troubleshooting)

---

## Tech Stack

- **Python 3.10** (recommended)
- **InsightFace** (`insightface==0.7.3`)
- **ONNX Runtime** (`onnxruntime` or `onnxruntime-gpu`)
- **NumPy**
- **OpenCV** (if used in your scripts)

---

## Project Structure (example)

> Update this section if your exact folder names differ.

```text
Infopoint_face_recognition/
├── Insight_Face_Recognition/
│   ├── insightface/
│   │   ├── app/
│   │   ├── commands/
│   │   ├── data/
│   │   │   ├── images/
│   │   │   └── objects/
│   │   ├── model_zoo/
│   │   ├── thirdparty/
│   │   │   └── face3d/
│   │   └── utils/
│   ├── scripts/                  # optional (your run scripts)
│   ├── requirements.txt          # optional
│   ├── README.md
│   └── ...
├── venv/                         # optional local virtual environment
└── ...
```

---

## Requirements

- **Python 3.10.x** (recommended)
- **64-bit Python** (important on Windows)
- Internet connection (for downloading packages/models on first run)

### Recommended Python Version
- ✅ Python **3.10**
- ⚠️ Python **3.12** may cause installation/build issues for some dependencies

---

## Installation

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd Infopoint_face_recognition-main/Insight_Face_Recognition
```

If you already downloaded the project manually, just open the project folder in terminal.

---

### 2) Create and activate a virtual environment (Python 3.10)

#### Windows (PowerShell)

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

#### Windows (CMD)

```cmd
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

#### Linux / macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

### 3) Install dependencies

#### CPU version (recommended for first setup)

```bash
python -m pip install onnxruntime
python -m pip install insightface==0.7.3
```

#### GPU version (optional)

```bash
python -m pip install onnxruntime-gpu
python -m pip install insightface==0.7.3
```

> Use the GPU version only if your CUDA/cuDNN setup is compatible.

---

### 4) Verify installation

```python
import insightface
import onnxruntime as ort

print("insightface:", insightface.__version__)
print("onnxruntime:", ort.__version__)
```

If this runs without errors, the setup is working.

---

## How to Run

Run your main script (replace with your actual entry file):

```bash
python main.py
```

or

```bash
python app.py
```

or

```bash
python recognize.py
```

> If your project uses webcam input, ensure your camera is connected and accessible.

---

## Typical Workflow (Face Recognition Projects)

1. **Prepare reference faces**
   - Store known persons' images in a folder (e.g., `data/images/known/`)
2. **Load recognition model**
   - Initialize InsightFace app/model
3. **Capture input**
   - From image / webcam / video
4. **Detect faces**
5. **Extract embeddings**
6. **Compare embeddings**
   - Match against known database
7. **Display / log result**
   - Show labels, timestamps, attendance records, etc.

---

## Troubleshooting

### `insightface` installation fails on Windows (`cl.exe` / build errors)

This usually means **Visual Studio Build Tools** are missing.

#### Fix
Install **Visual Studio Build Tools 2022** with:
- **Desktop development with C++**
- **MSVC v143** (or latest available MSVC toolset)
- **Windows 10/11 SDK**

Then retry:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install insightface==0.7.3
```

---

### Python version issue (3.12 / unsupported build problems)

If installation fails on **Python 3.12**, use **Python 3.10**.

#### Check Python version

```bash
python --version
```

#### Check Python architecture (must be 64-bit)

```bash
python -c "import struct; print(struct.calcsize('P') * 8)"
```

Expected output:

```text
64
```

---

### `onnxruntime` vs `onnxruntime-gpu`

- Use `onnxruntime` if:
  - you want simple setup
  - CPU is enough
- Use `onnxruntime-gpu` if:
  - you have an NVIDIA GPU
  - CUDA/cuDNN is configured correctly

If GPU install causes issues, uninstall and switch to CPU:

```bash
python -m pip uninstall -y onnxruntime-gpu
python -m pip install onnxruntime
```

---

### PowerShell blocks venv activation

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### Model download issues on first run

InsightFace may download model files automatically the first time you run it.

If download fails:
- check internet connection
- retry the script
- ensure firewall/proxy is not blocking Python

---

## Recommended `requirements.txt` (optional)

If you want a simple `requirements.txt`, use:

```txt
insightface==0.7.3
onnxruntime
numpy
opencv-python
tqdm
```

For GPU systems, replace `onnxruntime` with `onnxruntime-gpu`.

---

## Performance Tips

- **CPU mode** is good for development and testing.
- **GPU mode** can improve real-time performance significantly.
- For faster recognition:
  - resize frames before inference
  - run recognition every 2nd/3rd frame instead of every frame
  - cache embeddings for known identities

---

## Roadmap (Ideas for Improvement)

- Add attendance logging (CSV / SQLite / MySQL)
- Add GUI dashboard (Tkinter / Streamlit / Flask)
- Improve unknown-face threshold tuning
- Multi-camera support
- Better handling for occlusion / masks
- Docker setup for deployment
- REST API endpoint for recognition service

---

## Contributing

Contributions are welcome.

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## License

Add your license here (e.g., MIT / Apache-2.0).

Example:

```text
MIT License
```

---

## Author

**Samarth Patel**  
M.Sc. Automation & Robotics, TU Dortmund

