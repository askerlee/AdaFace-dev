import os
import argparse
import time
import tensorflow as tf
from eval_utils import compare_face_folders

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
                compare_face_folders(subdir_path, subdir_path)
                subdir_count += 1
            if subdir_count == 0:
                print("Self comparing %s:" %(args.path1))
                compare_face_folders(args.path1, args.path1)
        else:            
            compare_face_folders(args.path1, args.path2, verbose=False)
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
            compare_face_folders(subdir_path1, subdir_path2)        
        
    end = time.time()
    print("Time elapsed: %.2f seconds" %(end - begin))
