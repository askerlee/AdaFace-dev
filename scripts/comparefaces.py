from deepface import DeepFace
import sys
import os
import glob

path1 = sys.argv[1]
path2 = sys.argv[2]

if os.path.isfile(path1):
    img1_paths = [ path1 ]
else:
    img1_paths = glob.glob(path1 + "/*jpg")

if os.path.isfile(path2):
    img2_paths = [ path2 ]
else:
    img2_paths = glob.glob(path2 + "/*jpg")

for img1_path in img1_paths:
    for img2_path in img2_paths:
        print("%s vs %s" %(img1_path, img2_path))
        result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path)
        print(result)
        print()

