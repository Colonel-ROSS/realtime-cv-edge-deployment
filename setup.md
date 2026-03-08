# Raspberry Pi Setup Guide

## System Requirements

- Raspberry Pi 4 Model B (4GB RAM)
- Raspberry Pi OS (64-bit, Bullseye or later)
- Sony IMX219 camera module
- Python 3.9+

## Step 1 — Enable Camera Interface

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

## Step 2 — System Dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Build tools
sudo apt-get install -y build-essential cmake

# dlib dependencies
sudo apt-get install -y libopenblas-dev liblapack-dev

# OpenCV dependencies
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libatlas-base-dev libjasper-dev

# Picamera2 dependencies
sudo apt-get install -y python3-picamera2
```

## Step 3 — Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> ⚠️ **Note:** Installing dlib on Raspberry Pi takes 15–30 minutes due to compilation. This is expected.

## Step 4 — Prepare Known Faces

```bash
mkdir known_faces

# Create a subdirectory for each person
mkdir known_faces/YourName

# Add 5-10 images of each person
# Images should be:
# - Clear frontal or slight angle shots
# - Varied lighting conditions for robustness
# - JPG or PNG format
```

## Step 5 — Run the System

```bash
python src/main.py
```

On first run, face encodings will be computed and cached (~3 minutes).  
Subsequent runs will load from cache (< 1 minute).

## Troubleshooting

**Low FPS (< 10)?**
- Ensure no other processes are running
- Check system temperature: `vcgencmd measure_temp`
- If temp > 70°C, add cooling or reduce PROCESS_EVERY_N_FRAMES to 5

**Camera not detected?**
- Verify camera ribbon cable is fully seated
- Run: `libcamera-hello` to test camera independently

**Face recognition failures?**
- Check lighting — minimum 50 lux required for reliable detection
- Add more varied training images (different angles, lighting)
- Lower the recognition threshold in `face_utils.py`
