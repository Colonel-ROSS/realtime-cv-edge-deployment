#!/usr/bin/env python3
"""
Raspberry Pi FPS Benchmark
Runs YOLO + face_recognition for 20 seconds and reports true average FPS.
Run this BEFORE recording to calibrate FPS_PLAYBACK in main.py.

Usage:
    python simple_fps_test.py

Author: Musab Humzah Syed
"""

import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import face_recognition

print("=" * 50)
print("FPS BENCHMARK")
print("=" * 50)
print("\nRuns for 20 seconds under full YOLO + face_recognition load.")
print("Set up your typical recording scene before starting.\n")

TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
TEST_DURATION = 20

picam2 = Picamera2()
video_config = picam2.create_preview_configuration(
    main={"size": (TARGET_WIDTH, TARGET_HEIGHT)},
)
picam2.configure(video_config)
picam2.start()

print("Loading YOLO model...")
model = YOLO("yolov5n.pt")
print("✅ Model loaded\n")

frame_count = 0
start_time = time.time()

print(f"Starting {TEST_DURATION}s benchmark...")

while True:
    elapsed = time.time() - start_time
    if elapsed >= TEST_DURATION:
        break

    frame = picam2.capture_array()
    if frame is None:
        continue
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        results = model.predict(frame, imgsz=416, verbose=False)
        detection_count = len(results[0].boxes)
    except Exception:
        detection_count = 0

    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_count = len(face_locations)
    except Exception:
        face_count = 0

    frame_count += 1

    if frame_count % 10 == 0:
        current_fps = frame_count / elapsed
        print(f"  {elapsed:.1f}s | Frames: {frame_count} | FPS: {current_fps:.2f} | "
              f"Objects: {detection_count} | Faces: {face_count}")

total_time = time.time() - start_time
actual_fps = frame_count / total_time

print("\n" + "=" * 50)
print("BENCHMARK COMPLETE")
print("=" * 50)
print(f"\n  Duration:  {total_time:.2f}s")
print(f"  Frames:    {frame_count}")
print(f"  Avg FPS:   {actual_fps:.2f}")
print(f"\n  ⭐ Set FPS_PLAYBACK = {round(actual_fps, 2)} in main.py")

picam2.stop()
