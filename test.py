import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import cv2

gpu_id = 0
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=gpu_id, det_size=(512, 512))
face_image = Image.open("subjects-celebrity/jiffpom/aug27-2021.jpg")
face_info = face_analyzer.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
breakpoint()

face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
# face_emb: [512,]
face_emb = face_info['embedding']
