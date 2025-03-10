import cv2
import numpy as np
import sys

# Load video
video_path = sys.argv[1]
output_path0 = sys.argv[2]
num_frames = int(sys.argv[3])

cap = cv2.VideoCapture(video_path)

# Video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

frames = []
for i in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

cap.release()

if len(frames) == num_frames:
    for i, frame in enumerate(frames):
        output_path = output_path0 + f'_{i}.png'
        cv2.imwrite(output_path, frame)
    print(f'Frames extracted to {output_path0}_*.png')
else:
    print(f'Number of frames must be >= {num_frames}')
