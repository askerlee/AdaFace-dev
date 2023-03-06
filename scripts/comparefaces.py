from deepface import DeepFace
import sys
import os
import glob

def compare_paths(path1, path2, verbose=False):
    if os.path.isfile(path1):
        img1_paths = [ path1 ]
    else:
        img1_paths = glob.glob(path1 + "/*jpg") + glob.glob(path1 + "/*jpeg")

    if os.path.isfile(path2):
        img2_paths = [ path2 ]
    else:
        img2_paths = glob.glob(path2 + "/*jpg") + glob.glob(path2 + "/*jpeg")

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
    if len(sys.argv) < 3:
        print("Usage: python comparefaces.py path1 path2")
        exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]
    if path2 == '--self':
        subdirs = os.listdir(path1)
        for subdir in subdirs:
            subdir_path = os.path.join(path1, subdir)
            print("Self comparing %s:" %(subdir_path))
            compare_paths(subdir_path, subdir_path)
    else:            
        compare_paths(path1, path2, verbose=False)
