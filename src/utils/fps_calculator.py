#!/usr/bin/env python3
"""
FPS Calculator Utility
Computes the correct FPS_PLAYBACK value to fix video playback speed drift.

The problem: OpenCV's VideoWriter interprets the fps parameter as playback
rate, not capture rate. When the Pi captures at ~1 FPS but you set 30 FPS,
the video plays back 30x too fast.

Usage:
    python fps_calculator.py
    Enter your video file path when prompted.

Author: Musab Humzah Syed
"""

import cv2
import os

print("=" * 50)
print("FPS CALCULATOR")
print("=" * 50)

video_path = input("\nEnter path to your video file: ").strip()

if not os.path.exists(video_path):
    print(f"❌ File not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

current_duration = frame_count / fps if fps > 0 else 0

print(f"\n📊 Video Information:")
print(f"   Resolution:       {width}x{height}")
print(f"   Reported FPS:     {fps:.2f}")
print(f"   Total Frames:     {frame_count}")
print(f"   Current Duration: {current_duration:.2f} seconds")

print("\n" + "=" * 50)
print("CALCULATE CORRECT FPS")
print("=" * 50)
print("\nHow long did you actually record for? (seconds)")

try:
    actual_duration = float(input("Actual recording duration: ").strip())
except ValueError:
    print("❌ Invalid input")
    exit(1)

correct_fps = frame_count / actual_duration

print(f"\n✅ RESULTS:")
print(f"   Current playback: {current_duration:.2f}s")
print(f"   Actual recording: {actual_duration:.2f}s")
print(f"   Speed ratio:      {current_duration / actual_duration:.2f}x")
print(f"\n   ⭐ Set FPS_PLAYBACK = {correct_fps:.2f} in main.py")

if current_duration < actual_duration:
    print(f"\n📈 Video is playing {actual_duration / current_duration:.1f}x TOO FAST")
elif current_duration > actual_duration:
    print(f"\n📉 Video is playing {current_duration / actual_duration:.1f}x TOO SLOW")
else:
    print(f"\n✅ Video speed is correct!")
