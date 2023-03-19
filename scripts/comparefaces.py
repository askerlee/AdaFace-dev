from deepface import DeepFace
import sys
import os
import glob
import argparse
import time
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", type=str, help="path1")
    parser.add_argument("path2", type=str, nargs='?', default=None, help="path2")
    # --self
    parser.add_argument("--self", dest="self_comp", action="store_true", help="self comparison")
    # --pair
    parser.add_argument("--pair", dest="folderpair_comp", action="store_true", help="subfolder pairwise comparison")
    # --gpu
    parser.add_argument("--gpu", dest='gpu_id', type=int, default=0, help="specify a gpu to use")
    args = parser.parse_args()
    return args

def compare_paths(path1, path2, verbose=False):
    if os.path.isfile(path1):
        img1_paths = [ path1 ]
    else:
        img_extensions = [ "jpg", "jpeg", "png", "bmp" ]
        img1_paths = []
        for ext in img_extensions:
            img1_paths += glob.glob(path1 + "/*" + ext)

    if os.path.isfile(path2):
        img2_paths = [ path2 ]
    else:
        img_extensions = [ "jpg", "jpeg", "png", "bmp" ]
        img2_paths = []
        for ext in img_extensions:
            img2_paths += glob.glob(path2 + "/*" + ext)

    normal_pair_count = 0
    except_pair_count = 0
    total_distance = 0
    total_pair_count = len(img1_paths) * len(img2_paths)
    curr_pair_count = 0

    for img1_path in img1_paths:
        for img2_path in img2_paths:
            curr_pair_count += 1
            if img1_path == img2_path:
                continue
            img1_name = os.path.basename(img1_path)
            img2_name = os.path.basename(img2_path)
            if verbose:
                print("%d/%d: %s vs %s" %(curr_pair_count, total_pair_count, img1_name, img2_name))
            try:
                result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, 
                                         model_name="ArcFace", detector_backend = "retinaface")
            except:
                except_pair_count += 1
                continue

            distance = result['distance']
            total_distance += distance
            normal_pair_count += 1
            curr_avg_distance = total_distance / normal_pair_count
            if verbose:
                print("%.3f / %.3f" %(distance, curr_avg_distance))

    if normal_pair_count > 0:
        avg_distance = total_distance / normal_pair_count
    else:
        avg_distance = 0

    print("Normal pairs: %d, exception pairs: %d" %(normal_pair_count, except_pair_count))
    print("'%s' vs '%s' avg distance: %.3f" %(path1, path2, avg_distance))

if __name__ == "__main__":
    args = parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

    skip_subjs = [ 'corgi', 'lilbub', 'jiffpom', 'princessmonstertruck' ]

    begin = time.time()

    if not args.folderpair_comp:
        if args.self_comp:
            subdirs = sorted(os.listdir(args.path1))
            subdir_count = 0
            for subdir in subdirs:
                subdir_path = os.path.join(args.path1, subdir)
                if os.path.isdir(subdir_path) == False:
                    continue

                print("Self comparing %s:" %(subdir_path))
                compare_paths(subdir_path, subdir_path)
                subdir_count += 1
            if subdir_count == 0:
                print("Self comparing %s:" %(args.path1))
                compare_paths(args.path1, args.path1)
        else:            
            compare_paths(args.path1, args.path2, verbose=False)
    else:
        subdirs = sorted(os.listdir(args.path1))
        for subdir in subdirs:
            if subdir in skip_subjs:
                continue
            subdir_path1 = os.path.join(args.path1, subdir)
            subdir_path2 = os.path.join(args.path2, subdir)
            if os.path.isdir(subdir_path1) == False or os.path.isdir(subdir_path2) == False:
                continue
            print("Pair comparing %s vs %s:" %(subdir_path1, subdir_path2))
            compare_paths(subdir_path1, subdir_path2)        
        
    end = time.time()
    print("Time elapsed: %.2f seconds" %(end - begin))
