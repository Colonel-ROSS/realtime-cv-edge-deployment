# Real-Time Multi-Person Face Recognition & Object Detection on Edge Hardware

> Deployed on Raspberry Pi 4 (4GB) using YOLOv5n + dlib ResNet embeddings.  
> Optimised from **5–8 FPS → 15–25 FPS** through systematic edge deployment techniques.

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green?style=flat-square)
![YOLOv5](https://img.shields.io/badge/YOLOv5n-Ultralytics-red?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-c51a4a?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Problem Statement

Running simultaneous face recognition and object detection on resource-constrained hardware is a hard problem. Most CV systems assume GPU availability. This project answers a different question:

**How do you deploy a real-time, multi-person CV pipeline on a €35 computer — and actually make it work?**

This system detects and identifies multiple people entering, exiting, and re-entering a dynamic scene while simultaneously detecting objects — all running live on a Raspberry Pi 4 with no GPU acceleration.

---

## 🎬 Demo

> *(Add demo GIF here — record a short screen capture of the system running and convert to GIF using ezgif.com)*

```
[DEMO GIF PLACEHOLDER]
```

**System in action:** Bounding boxes around detected faces with confidence scores, YOLO object labels, and real-time FPS counter overlaid on the video stream.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4 (4GB)                  │
│                                                          │
│  Sony IMX219 Camera                                      │
│       │                                                  │
│       ▼                                                  │
│  Picamera2 Frame Capture (640×480)                       │
│       │                                                  │
│       ▼                                                  │
│  ┌────────────────┐    ┌─────────────────────────────┐  │
│  │  Every N=3     │    │     Cached Results           │  │
│  │  frames:       │    │     (intermediate frames)    │  │
│  │                │    └─────────────────────────────┘  │
│  │  ┌──────────┐  │                                      │
│  │  │ YOLOv5n  │  │  ← 416×416 input, 3.9MB model       │
│  │  │ Object   │  │    ~120-150ms inference               │
│  │  │ Detection│  │                                      │
│  │  └──────────┘  │                                      │
│  │                │                                      │
│  │  ┌──────────┐  │                                      │
│  │  │  dlib    │  │  ← Half-resolution input (0.5x)      │
│  │  │  ResNet  │  │    128-D face embeddings              │
│  │  │  Face ID │  │    Euclidean distance matching        │
│  │  └──────────┘  │                                      │
│  └────────────────┘                                      │
│       │                                                  │
│       ▼                                                  │
│  Annotated Frame + CSV Logging                           │
│  (movement_log.csv + object_log.csv)                     │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance Results

### Optimisation Impact

| Metric | Before Optimisation | After Optimisation |
|--------|--------------------|--------------------|
| Frame Rate | 5–8 FPS | **15–25 FPS** |
| Processing Lag | 2–3 seconds | < 0.1 seconds |
| System Temp | > 70°C | Stable |
| Startup Time | ~3 minutes | **< 1 minute** (cached encodings) |

### Detection Performance by Lighting Condition

| Scenario | Lighting | Face Recognition Confidence | Object Detection Rate |
|----------|----------|----------------------------|-----------------------|
| Normal | 150–250 lux | 40–60% | 70–85% (common objects) |
| Low Light | < 50 lux | 35–50% | 30–40% drop |
| Directional | 115–130 lux | ~58% | Variable (shadow false positives) |

**Key finding:** 50 lux is the critical threshold below which system reliability degrades significantly — driven by sensor physics (analog gain amplifying noise), not algorithmic limitations.

---

## 🔧 Optimisation Techniques

### 1. Frame-Skip Processing
Instead of running inference on every frame (impossible at 30 FPS on ARM CPU), the system processes every N=3 frames and caches detection results for intermediate frames. This reduces inference frequency by **66%** while maintaining smooth visual output.

### 2. Multi-Resolution Face Detection
Face recognition operations are O(n²) with image dimensions. The system resizes frames to **0.5× resolution** for detection, then scales bounding box coordinates back to original size. Result: significantly faster processing with minimal accuracy loss.

### 3. Encoding Cache (MD5 Hashing)
Face encodings (128-D embeddings) are computed once and stored in a pickle file. An MD5 hash of the known_faces directory detects changes. Subsequent runs load from cache, reducing startup from **~3 minutes to under 1 minute**.

### 4. Model Selection: YOLOv5n over YOLOv5s
| Model | Size | Inference Time | mAP Trade-off |
|-------|------|---------------|---------------|
| YOLOv5s | 27 MB | ~400ms/frame | Baseline |
| **YOLOv5n** | **3.9 MB** | **~120-150ms/frame** | -5–7% mAP |

The 3–4× inference speedup is worth the small accuracy trade-off for real-time edge deployment.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9 |
| Camera Interface | Picamera2 v0.3.12 |
| Object Detection | YOLOv5n (Ultralytics) |
| Face Recognition | face_recognition v1.3.0 (dlib ResNet-34) |
| Image Processing | OpenCV 4.8.0 |
| Data Logging | CSV + Pandas |
| Hardware | Raspberry Pi 4 Model B (4GB) |
| Camera Sensor | Sony IMX219 8MP |

---

## 📁 Repository Structure

```
realtime-cv-edge-deployment/
│
├── src/
│   ├── main.py                    # Main detection + recognition pipeline
│   ├── face_utils.py              # Encoding cache, face matching logic
│   └── object_detection.py        # YOLO inference wrapper
│
├── docs/
│   └── architecture.png           # System architecture diagram
│
├── results/
│   ├── movement_log.csv           # Sample: person entry/exit timestamps + confidence
│   └── object_log.csv             # Sample: object detections over time
│
├── known_faces/                   # Directory for known face images (gitignored)
├── requirements.txt               # Python dependencies
├── setup.md                       # Raspberry Pi setup guide
└── README.md
```

---

## 🚀 Setup & Installation

### Hardware Requirements
- Raspberry Pi 4 Model B (4GB RAM recommended)
- Sony IMX219 camera module (or compatible CSI camera)
- MicroSD card (32GB+, Class 10)

### Software Setup

```bash
# Clone the repository
git clone https://github.com/Colonel-ROSS/realtime-cv-edge-deployment.git
cd realtime-cv-edge-deployment

# Install dependencies
pip install -r requirements.txt

# Add known face images
mkdir known_faces
# Add subdirectories named after each person with 5-10 images each
# known_faces/PersonName/img1.jpg, img2.jpg ...

# Run the system
python src/main.py
```

### Runtime Controls

| Key | Action |
|-----|--------|
| `O` / `P` | Decrease / Increase brightness (range: 0.1–2.5) |
| `K` / `L` | Decrease / Increase sharpness |
| `Q` | Quit and save logs |

---

## 📊 Data Logging

The system generates two CSV logs automatically:

**movement_log.csv** — Person-level temporal data
```
name, entry_time, exit_time, avg_confidence
Musab, 00:00:02.1, 00:00:18.4, 0.54
```

**object_log.csv** — Object detection events
```
object_class, confidence, timestamp
chair, 0.72, 00:00:01.3
cup, 0.61, 00:00:03.7
```

These logs enable post-session analysis: presence timelines, confidence trends, and object detection frequency distributions.

---

## 🧠 Key Technical Learnings

- **Lighting is the dominant variable** in both face recognition and object detection performance — more impactful than model choice or resolution settings
- **Temporal correlation in video** can be exploited for performance: caching detection results across frames trades marginal accuracy for significant throughput gains
- **Edge deployment forces engineering discipline** — every millisecond matters, and naive implementations that work on a laptop will fail on ARM hardware
- **ISP pipeline awareness matters** — understanding how the camera's Image Signal Processor handles gain, noise reduction, and white balance directly informs optimisation strategy

---

## 📄 Academic Context

This project was developed as part of the **CE6461 – Computer Vision Systems** module at the **University of Limerick** (MEng Computer Vision & AI, 2025–2026).

---

## 👤 Author

**Musab Humzah Syed**  
MEng Computer Vision & AI — University of Limerick  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/musab-humzah-syed/)
[![Medium](https://img.shields.io/badge/Medium-Blog-black?style=flat-square&logo=medium)](https://medium.com/@musabhumzah.2002)

---

## 📜 License

MIT License — feel free to use, adapt, and build on this work.
