#!/usr/bin/env python3
"""
Plot Generator — CS6461 Computer Vision Systems
Generates 4 analysis plots from movement_log.csv and object_log.csv
produced by main.py after a recording session.

Output files:
  - timeline_plot.png         → people & object presence over time
  - confidence_plot.png       → face recognition confidence over time
  - object_confidence_plot.png → object detection confidence over time
  - summary_statistics.png    → 2×2 summary panel

Usage:
    python plot_generator.py

Author: Musab Humzah Syed
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# Load CSV logs
# ─────────────────────────────────────────────
movement_df = pd.read_csv('movement_log.csv')
object_df = pd.read_csv('object_log.csv')

people = movement_df['Name'].unique()
objects = object_df['Object'].unique()
colors_people = plt.cm.Set3(np.linspace(0, 1, len(people)))
colors_objects = plt.cm.Paired(np.linspace(0, 1, len(objects)))

# ─────────────────────────────────────────────
# Plot 1: Detection Timelines
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for idx, person in enumerate(people):
    person_data = movement_df[movement_df['Name'] == person]
    for _, row in person_data.iterrows():
        entry = row['Entry Time (s)']
        exit_time = row['Exit Time (s)']
        duration = exit_time - entry
        ax1.barh(idx, duration, left=entry, height=0.8,
                 color=colors_people[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.text(entry + duration / 2, idx, f'{row["Avg Confidence"]:.2f}',
                 ha='center', va='center', fontweight='bold', fontsize=9)

ax1.set_yticks(range(len(people)))
ax1.set_yticklabels(people, fontsize=11)
ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('People Detected', fontsize=12, fontweight='bold')
ax1.set_title('Timeline of People Detection with Average Confidence',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim(0, movement_df['Exit Time (s)'].max() + 1)

max_time = object_df['Timestamp (s)'].max()
for idx, obj in enumerate(objects):
    obj_data = object_df[object_df['Object'] == obj]
    for t in range(int(max_time) + 1):
        obj_at_time = obj_data[(obj_data['Timestamp (s)'] >= t) & (obj_data['Timestamp (s)'] < t + 1)]
        if len(obj_at_time) > 0:
            ax2.barh(idx, 1, left=t, height=0.8,
                     color=colors_objects[idx], alpha=0.7, edgecolor='black', linewidth=1)

ax2.set_yticks(range(len(objects)))
ax2.set_yticklabels(objects, fontsize=11)
ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Objects Detected', fontsize=12, fontweight='bold')
ax2.set_title('Timeline of Object Detection', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, max_time + 1)

plt.tight_layout()
plt.savefig('timeline_plot.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved timeline_plot.png")

# ─────────────────────────────────────────────
# Plot 2: Face Confidence Over Time
# ─────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(14, 8))

for person in people:
    person_entries = movement_df[movement_df['Name'] == person]
    for i, (_, row) in enumerate(person_entries.iterrows()):
        ax3.plot([row['Entry Time (s)'], row['Exit Time (s)']],
                 [row['Avg Confidence'], row['Avg Confidence']],
                 linewidth=3, marker='o', markersize=8,
                 label=person if i == 0 else "")

ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
ax3.set_title('Face Recognition Confidence Over Time', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, 1)
ax3.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('confidence_plot.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved confidence_plot.png")

# ─────────────────────────────────────────────
# Plot 3: Object Detection Confidence Over Time
# ─────────────────────────────────────────────
fig3, ax4 = plt.subplots(figsize=(12, 6))

for obj in objects:
    obj_data = object_df[object_df['Object'] == obj]
    ax4.plot(obj_data['Timestamp (s)'], obj_data['Confidence'],
             marker='o', linewidth=2, markersize=4, alpha=0.7, label=obj)

ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Detection Confidence', fontsize=12, fontweight='bold')
ax4.set_title('Object Detection Confidence Over Time', fontsize=14, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='best', fontsize=10)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('object_confidence_plot.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved object_confidence_plot.png")

# ─────────────────────────────────────────────
# Plot 4: Summary Statistics (2×2 panel)
# ─────────────────────────────────────────────
fig4, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(14, 10))

person_avg_conf = movement_df.groupby('Name')['Avg Confidence'].mean()
ax5.bar(range(len(person_avg_conf)), person_avg_conf.values,
        color=colors_people[:len(person_avg_conf)], alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(person_avg_conf)))
ax5.set_xticklabels(person_avg_conf.index, rotation=45, ha='right')
ax5.set_ylabel('Average Confidence', fontweight='bold')
ax5.set_title('Avg Face Recognition Confidence by Person', fontweight='bold')
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.set_ylim(0, 1)

person_time = movement_df.groupby('Name').apply(
    lambda x: (x['Exit Time (s)'] - x['Entry Time (s)']).sum()
)
ax6.bar(range(len(person_time)), person_time.values,
        color=colors_people[:len(person_time)], alpha=0.7, edgecolor='black')
ax6.set_xticks(range(len(person_time)))
ax6.set_xticklabels(person_time.index, rotation=45, ha='right')
ax6.set_ylabel('Total Time (seconds)', fontweight='bold')
ax6.set_title('Total Scene Presence by Person', fontweight='bold')
ax6.grid(axis='y', alpha=0.3, linestyle='--')

object_counts = object_df['Object'].value_counts()
ax7.bar(range(len(object_counts)), object_counts.values,
        color=colors_objects[:len(object_counts)], alpha=0.7, edgecolor='black')
ax7.set_xticks(range(len(object_counts)))
ax7.set_xticklabels(object_counts.index, rotation=45, ha='right')
ax7.set_ylabel('Detection Count', fontweight='bold')
ax7.set_title('Object Detection Frequency', fontweight='bold')
ax7.grid(axis='y', alpha=0.3, linestyle='--')

object_avg_conf = object_df.groupby('Object')['Confidence'].mean()
ax8.bar(range(len(object_avg_conf)), object_avg_conf.values,
        color=colors_objects[:len(object_avg_conf)], alpha=0.7, edgecolor='black')
ax8.set_xticks(range(len(object_avg_conf)))
ax8.set_xticklabels(object_avg_conf.index, rotation=45, ha='right')
ax8.set_ylabel('Average Confidence', fontweight='bold')
ax8.set_title('Average Object Detection Confidence', fontweight='bold')
ax8.grid(axis='y', alpha=0.3, linestyle='--')
ax8.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('summary_statistics.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved summary_statistics.png")

plt.show()
print("[INFO] All plots generated successfully!")
