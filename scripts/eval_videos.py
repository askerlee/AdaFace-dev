import os
import argparse
import time
from evaluation.eval_utils import compare_face_folders, set_tf_gpu
import cv2, numpy as np, glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Root path")
    parser.add_argument("--subject", type=str, default=None, help="Subject path")
    parser.add_argument("--gpu", dest='gpu_id', type=int, default=0, help="specify a gpu to use")
    parser.add_argument("--face_engine", dest='face_engine', type=str, default='deepface', 
                        choices=['deepface', 'insightface'],
                        help="face engine to use for comparison")
    args = parser.parse_args()
    return args

def extract_frames(video_path, interval=1, collate=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_count += 1
        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    if collate:
        frames = np.array(frames)

    cap.release()
    return frames

if __name__ == "__main__":
    args = parse_args()

    set_tf_gpu(args.gpu_id)
    img_extensions   = [ "jpg", "jpeg", "png", "bmp" ]
    video_extensions = [ "mp4" ]
    simi_stats = { 'adaface': [], 'idani': [], 'luma': [] }
    begin = time.time()

    subject_paths = []

    if args.root is not None:
        subdirs = sorted(os.listdir(args.root))
        for subject in subdirs:
            subject_path = os.path.join(args.root, subject)
            if os.path.isdir(subject_path) == False:
                continue
            subject_paths.append(subject_path)

    if args.subject is not None:
        subject_paths.append(args.subject)

    subject_count = 0
    video_count = 0

    for subject_path in subject_paths:
        print(subject_path)

        image_filenames = []
        video_filenames = []
        subj_simi_stats = { 'adaface': [], 'idani': [], 'luma': [] }

        for filename in sorted(os.listdir(subject_path)):
            for ext in img_extensions:
                if filename.endswith(ext) and '-init.' not in filename:
                    image_filenames.append(os.path.join(subject_path, filename))
                    break
            for ext in video_extensions:
                if filename.endswith(ext):
                    video_filenames.append(os.path.join(subject_path, filename))
                    break

        if len(image_filenames) == 0 or len(video_filenames) == 0:
            print(f"Skipping {subject_path} due to lack of images or videos")
            continue

        for video_filename in video_filenames:
            print("Processing %s" %video_filename)
            frames = extract_frames(video_filename, interval=3, collate=False)
            avg_similarity, dst_normal_img_count, dst_no_face_img_count = \
                compare_face_folders(image_filenames, frames, face_engine=args.face_engine, 
                                        cache_src_embeds=False, verbose=False)
            
            for key in simi_stats:
                if video_filename.endswith(f"-{key}.mp4"):
                    simi_stats[key].append(avg_similarity)
                    subj_simi_stats[key].append(avg_similarity)

            print(f"Avg sim on {dst_normal_img_count} frames: {avg_similarity:.2f}")
            video_count += 1

        subject_count += 1
        print(f"{subject_path} stats:")
        for key in subj_simi_stats:
            avg_similarity = np.mean(subj_simi_stats[key])
            print(f"{key}: {avg_similarity:.3f}")

    print(f"Total {subject_count} subjects and {video_count} videos processed")
    print("Average similarity:")
    for key in simi_stats:
        avg_similarity = np.mean(simi_stats[key])
        print(f"{key}: {avg_similarity:.3f}")

    end = time.time()
    print("Time elapsed: %.2f seconds" %(end - begin))
