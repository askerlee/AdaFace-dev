import os
import argparse
import time
from evaluation.eval_utils import compare_face_folders, set_tf_gpu
import cv2, numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Root path")
    parser.add_argument("--subject", type=str, default=None, help="Subject path")
    parser.add_argument("--methods", type=str, nargs='*', default=None,
                        choices=['adaface', 'idani', 'luma', 'pika'], help="Select methods to evaluate")
    parser.add_argument("--single_video", type=str, default=None, help="Single video for evaluation")
    parser.add_argument("--ref_image", type=str, default=None, help="Reference image for evaluation")
    parser.add_argument("--gpu", dest='gpu_id', type=int, default=0, help="specify a gpu to use")
    parser.add_argument("--face_engine", dest='face_engine', type=str, default='deepface', 
                        choices=['deepface', 'insightface'],
                        help="face engine to use for comparison")
    parser.add_argument("--sample_interval", type=int, default=3,
                        help="Sample interval for video frames")
    parser.add_argument("--save_frames_to", type=str, default=None,
                        help="Save frames to this directory")
    parser.add_argument("--verbose", dest='verbose', action='store_true', help="Verbose mode")
    parser.add_argument("--debug", dest='debug', action='store_true', help="Debug mode")
    args = parser.parse_args()
    return args

# collate: whether to collate the frames into a single numpy array.
def extract_frames(video_path, sample_interval=1, collate=False, save_frames_to=None):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_count += 1
        if frame_count % sample_interval == 0:
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
    simi_stats = { 'adaface': [], 'idani': [], 'luma': [], 'pika': [] }
    begin = time.time()

    subject_paths = []

    if args.root is not None:
        subdirs = sorted(os.listdir(args.root))
        for subject in subdirs:
            subject_path = os.path.join(args.root, subject)
            if os.path.isdir(subject_path) == False:
                continue
            # Ignore subjects with ignore.txt in their folder.
            # These are private subjects that should not be counted in the public dataset.
            if os.path.exists(os.path.join(subject_path, "ignore.txt")):
                continue
            subject_paths.append(subject_path)

    if args.subject is not None:
        subject_paths.append(args.subject)

    subject_count = 0
    video_count = 0

    if args.single_video is not None:
        assert args.ref_image is not None, "Reference image must be provided"
        frames = extract_frames(args.single_video, sample_interval=args.sample_interval, collate=False)
        print(f"Extracted {len(frames)} frames from {args.single_video}")
        if args.save_frames_to is not None:
            os.makedirs(args.save_frames_to, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(args.save_frames_to, f"frame-{i:04d}.jpg"))
            print(f"Saved {len(frames)} frames to {args.save_frames_to}")

        print(f"Processing {args.single_video} ({len(frames)} frames)")
        all_similarities, avg_similarity, normal_frame_count, no_face_frame_count = \
            compare_face_folders([args.ref_image], frames, face_engine=args.face_engine,
                                    cache_src_embeds=False, verbose=True)
        print(f"Avg sim on {normal_frame_count} frames ({no_face_frame_count} no face): {avg_similarity:.3f}")
        # By default, in the single_video mode, all similarities will be printed for diagnostic purposes.
        args.verbose = True
        if args.verbose:
            for i, simi in enumerate(all_similarities):
                if i == len(all_similarities) - 1 or (i > 0 and i % 10 == 0):
                    end_char = "\n"
                else:
                    end_char = ", "
                print(f"{simi:.3f}", end=end_char)
                        
        exit()

    for subject_path in subject_paths:
        print(subject_path)

        image_filenames = []
        video_filenames = []
        subj_simi_stats = { 'adaface': [], 'idani': [], 'luma': [], 'pika': [] }

        for filename in sorted(os.listdir(subject_path)):
            for ext in img_extensions:
                if filename.endswith(ext) and '-init.' not in filename:
                    image_filenames.append(os.path.join(subject_path, filename))
                    break
            for ext in video_extensions:
                if filename.endswith(ext):
                    if args.methods is not None:
                        for method in args.methods:
                            if filename.endswith(f"-{method}.mp4"):
                                video_filenames.append(os.path.join(subject_path, filename))
                                break
                    else:
                        video_filenames.append(os.path.join(subject_path, filename))
                        break

        if len(image_filenames) == 0 or len(video_filenames) == 0:
            print(f"Skipping {subject_path} due to lack of images or videos")
            continue

        for video_filename in video_filenames:
            print("Processing %s" %video_filename)
            frames = extract_frames(video_filename, sample_interval=args.sample_interval, collate=False)
            all_similarities, avg_similarity, normal_frame_count, no_face_frame_count = \
                compare_face_folders(image_filenames, frames, face_engine=args.face_engine, 
                                     cache_src_embeds=False, verbose=False, debug=args.debug)
            
            method = "unknown"
            for key in simi_stats:
                if video_filename.endswith(f"-{key}.mp4"):
                    method = key
                    simi_stats[key].append(avg_similarity)
                    subj_simi_stats[key].append(avg_similarity)
                    break

            print(f"{method:<7} sim on {normal_frame_count} frames ({no_face_frame_count} no face): {avg_similarity:.3f}")
            if args.verbose:
                for i, simi in enumerate(all_similarities):
                    if i == len(all_similarities) - 1 or (i > 0 and i % 10 == 0):
                        end_char = "\n"
                    else:
                        end_char = ", "
                    print(f"{simi:.3f}", end=end_char)

            video_count += 1

        subject_count += 1
        print(f"{subject_path} stats:")
        for key in subj_simi_stats:
            avg_similarity = np.mean(subj_simi_stats[key])
            print(f"{key}: {avg_similarity:.3f}")

    print(f"Total {subject_count} subjects and {video_count} videos processed")
    print("Average similarity/std:")
    for key in simi_stats:
        avg_similarity = np.mean(simi_stats[key])
        std_similarity = np.std(simi_stats[key])
        print(f"{key}: {avg_similarity:.3f}/{std_similarity:.3f}")

    end = time.time()
    print("Time elapsed: %.1f seconds" %(end - begin))
