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
                        choices=['adaface', 'idani', 'luma', 'pika', 'consist', 'arc2face'], 
                        help="Select methods to evaluate")
    parser.add_argument("--single_video", type=str, default=None, help="Single video for evaluation")
    parser.add_argument("--ref_image", type=str, default=None, help="Reference image for evaluation")
    parser.add_argument("--gpu", dest='gpu_id', type=int, default=0, help="specify a gpu to use")
    parser.add_argument("--face_engine", dest='face_engine', type=str, default='deepface', 
                        choices=['deepface', 'insightface'],
                        help="Face engine to use for comparison")
    parser.add_argument("--face_model", dest='face_model', type=str, default='ArcFace',
                        choices=['VGG-Face', 'ArcFace'],
                        help="Face model to use for comparison")
    parser.add_argument("--sample_interval", type=int, default=3,
                        help="Sample interval for video frames")
    parser.add_argument("--max_seconds_of_frames", type=int, default=-1,
                        help="First n seconds of frames to sample from the video (-1 for all)")
    parser.add_argument("--save_frames_to", type=str, default=None,
                        help="Save frames to this directory")
    parser.add_argument("--cmp_frames", dest='cmp_frames', type=int, nargs='+', default=None,
                        help="Compare frame 0 and frame i")
    parser.add_argument("--motion_ratio_threshold", type=float, default=0.2,
                        help="Threshold for face bboxes motion ratios in video frames for --cmp_frames")
    parser.add_argument("--sel_video_list", type=str, default='sel-video-list.txt',
                        help="File with list of videos to process")
    parser.add_argument("--apply_sel_video_list", action='store_true',
                        help="Apply the sel_video_list file to filter videos")
    parser.add_argument("--update_sel_video_list", action='store_true',
                        help="Update the sel_video_list file with the selected videos")
    parser.add_argument("--verbose", dest='verbose', action='store_true', help="Verbose mode")
    parser.add_argument("--debug", dest='debug', action='store_true', help="Debug mode")
    args = parser.parse_args()
    return args

# collate: whether to collate the frames into a single numpy array.
def extract_frames(video_path, sample_interval=1, collate=False, save_frames_to=None, 
                   max_seconds_of_frames=-1, verbose=True):
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if max_seconds_of_frames > 0:
        num_frames = int(fps * max_seconds_of_frames / sample_interval) + 1
        if len(frames) > num_frames:
            frames = frames[:num_frames]

    cap.release()

    if verbose:
        print(f"{video_path} total frames {frame_count}, fps {fps:.2f}. Extracted {len(frames)} frames")
    return frames

def calc_bbox_motion(bboxes1, bboxes2):
    if bboxes1 is None or bboxes2 is None or len(bboxes1) == 0 or len(bboxes2) == 0:
        return -1, -1
    # Calculate the motion between two bboxes
    # bbox1 and bbox2 are lists of [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = bboxes1[0]
    x1_2, y1_2, x2_2, y2_2 = bboxes2[0]
    # motion: Average motion of the 4 corners of the bounding box.
    motion = (abs(x1_1 - x1_2) + abs(y1_1 - y1_2) + abs(x2_1 - x2_2) + abs(y2_1 - y2_2)) / 4.0
    # motion_ratio: motion / bounding box edge length
    motion_ratio = motion * 2 / (abs(x2_1 - x1_1) + abs(y2_1 - y1_1))
    return motion, motion_ratio

if __name__ == "__main__":
    args = parse_args()

    set_tf_gpu(args.gpu_id)
    img_extensions   = [ "jpg", "jpeg", "png", "bmp" ]
    video_extensions = [ "mp4" ]
    simi_stats = { 'adaface': [], 'idani': [], 'luma': [], 'pika': [], 'consist': [], 'arc2face': [] }
    if args.methods is None:
        args.methods = simi_stats.keys()
    begin = time.time()

    subject_paths = []
    if args.cmp_frames is not None:
        if len(args.cmp_frames) == 2:
            # 0 5 -> 0 5 6 (6 not inclusive, so it's only frame 5)
            args.cmp_frames = args.cmp_frames + [args.cmp_frames[-1] + 1]
        assert len(args.cmp_frames) == 3, "cmp_frames must be 2 or 3 integers"

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
        frames = extract_frames(args.single_video, sample_interval=args.sample_interval, collate=False,
                                max_seconds_of_frames=args.max_seconds_of_frames, verbose=True)
        if args.save_frames_to is not None:
            os.makedirs(args.save_frames_to, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(args.save_frames_to, f"frame-{i:04d}.jpg"))
            print(f"Saved {len(frames)} frames to {args.save_frames_to}")

        print(f"Processing {args.single_video} ({len(frames)} frames)")
        all_similarities, avg_similarity, normal_frame_count, no_face_frame_count, list_face_bboxes = \
            compare_face_folders([args.ref_image], frames, face_engine=args.face_engine,
                                 face_model=args.face_model, cache_src_embeds=False, verbose=True)
        
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

    if args.apply_sel_video_list:
        if not os.path.exists(args.sel_video_list):
            print(f"Selected video list {args.sel_video_list} does not exist")
            exit()
        with open(args.sel_video_list, 'r') as f:
            sel_video_list = [line.strip() for line in f.readlines()]
        print(f"Selected video list {args.sel_video_list} contains {len(sel_video_list)} videos")
    else:
        sel_video_list = None
        
    new_sel_video_list = []
    for subject_path in subject_paths:
        print(subject_path)

        image_filenames = []
        video_filenames = []
        subj_simi_stats = { 'adaface': [], 'idani': [], 'luma': [], 'pika': [], 'consist': [], 'arc2face': [] }

        for filename in sorted(os.listdir(subject_path)):
            for ext in img_extensions:
                if filename.endswith(ext) and '-init.' not in filename:
                    image_filenames.append(os.path.join(subject_path, filename))
                    break
            for ext in video_extensions:
                if filename.endswith(ext):
                    for method in args.methods:
                        if filename.endswith(f"-{method}.mp4"):
                            video_filenames.append(os.path.join(subject_path, filename))
                            break

        if len(image_filenames) == 0 or len(video_filenames) == 0:
            print(f"Skipping {subject_path} due to lack of images or videos")
            continue

        for video_filename in video_filenames:
            if sel_video_list is not None:
                video_filename_nomethod = video_filename.split("-")[:-1]
                video_filename_nomethod = "-".join(video_filename_nomethod)
                if video_filename_nomethod not in sel_video_list:
                    print(f"Skipping {video_filename} not in {args.sel_video_list}")
                    continue
                
            print("Processing %s" %video_filename)
            frames = extract_frames(video_filename, sample_interval=args.sample_interval, collate=False,
                                    max_seconds_of_frames=args.max_seconds_of_frames, verbose=True)
            # all_similarities only contains similarity scores when faces are detected.
            # If max_seconds_of_frames > 0, avg_similarity is the average of the first max_seconds_of_frames frames.
            all_similarities, avg_similarity, normal_frame_count, no_face_frame_count, list_face_bboxes = \
                compare_face_folders(image_filenames, frames, face_engine=args.face_engine, 
                                     face_model=args.face_model, cache_src_embeds=False, verbose=False, debug=args.debug)

            filtered = False
            if args.cmp_frames is not None:
                cmp_i, cmp_j1, cmp_j2 = args.cmp_frames
                max_j_similarity = max(all_similarities[cmp_j1:cmp_j2])
                avg_similarity = max_j_similarity - all_similarities[cmp_i]
                print(f"Frame {cmp_i} vs Frames {cmp_j1}-{cmp_j2} similarity changes: {avg_similarity:.3f}")
                begin_frame_index = cmp_i
                end_frame_indices = list(range(cmp_j1, cmp_j2))
            else:
                # begin_frame_index is the first frame with a face detected.
                for begin_frame_index in range(len(all_similarities)):
                    if all_similarities[begin_frame_index] > 0:
                        break
                # all frames except the first one
                end_frame_indices = list(range(begin_frame_index + 1, len(all_similarities)))
            
            if len(end_frame_indices) == 0:
                print("No end frame indices found, discarding this video")
                filtered = True
            else:
                bbox_motions_and_ratios = [ calc_bbox_motion(list_face_bboxes[begin_frame_index], list_face_bboxes[j]) for j in end_frame_indices ]
                bbox_motions, bbox_motion_ratios = zip(*bbox_motions_and_ratios)
                max_bbox_motion         = max(bbox_motions)
                max_bbox_motion_ratio   = max(bbox_motion_ratios)
                print(f"motion: {max_bbox_motion:.1f} / {max_bbox_motion_ratio:.2f}.", end=" ")
                if (max_bbox_motion_ratio < 0) or (max_bbox_motion_ratio > args.motion_ratio_threshold):
                    filtered = True
                    print(f"Filtered by motion ratio {args.motion_ratio_threshold:.2f}")
                else:
                    print()

            if not filtered:
                video_filename_nomethod = video_filename.split("-")[:-1]
                video_filename_nomethod = "-".join(video_filename_nomethod)
                new_sel_video_list.append(video_filename_nomethod)

            method = "unknown"
            for key in simi_stats:
                if video_filename.endswith(f"-{key}.mp4"):
                    method = key
                    if not filtered:
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
            if key not in args.methods:
                continue

            avg_similarity = np.mean(subj_simi_stats[key])
            print(f"{key}: {avg_similarity:.3f}")

    print(f"Total {subject_count} subjects and {video_count} videos processed")
    print("Average similarity/std:")
    for key in simi_stats:
        if key not in args.methods:
            continue
        avg_similarity = np.mean(simi_stats[key])
        std_similarity = np.std(simi_stats[key])
        print(f"{key}: {len(simi_stats[key])} videos, similarity: {avg_similarity:.3f}/{std_similarity:.3f}")

    end = time.time()
    print("Time elapsed: %.1f seconds" %(end - begin))
    if args.update_sel_video_list and len(new_sel_video_list) > 0:
        with open(args.sel_video_list, 'w') as f:
            for video in new_sel_video_list:
                f.write(video + "\n")
        print(f"Updated {args.sel_video_list} with {len(new_sel_video_list)} videos")
